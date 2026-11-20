use crate::serve::engine::{ChatJob, Engine, Event};
use crate::serve::hpack::{self, HpackCtx};
use crate::serve::protocol::{
    ChatRequest, CompletionMeta, CompletionRequest, Delta, metrics_json, models_json, push_sse,
    push_sse_data,
};
use crate::serve::{Limits, Pump, pump_events};
use anyhow::{Result, anyhow};
use serde_json::json;
use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;

pub const PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

const FRAME_DATA: u8 = 0x0;
const FRAME_HEADERS: u8 = 0x1;
const FRAME_RST: u8 = 0x3;
const FRAME_SETTINGS: u8 = 0x4;
const FRAME_PING: u8 = 0x6;
const FRAME_GOAWAY: u8 = 0x7;
const FRAME_WINDOW_UPDATE: u8 = 0x8;
const FRAME_CONTINUATION: u8 = 0x9;

const FLAG_END_STREAM: u8 = 0x1;
const FLAG_ACK: u8 = 0x1;
const FLAG_END_HEADERS: u8 = 0x4;
const FLAG_PADDED: u8 = 0x8;
const FLAG_PRIORITY: u8 = 0x20;

const SETTINGS_HEADER_TABLE_SIZE: u16 = 0x1;
const SETTINGS_INITIAL_WINDOW_SIZE: u16 = 0x2;

const ERR_PROTOCOL: u32 = 0x1;
const ERR_FLOW_CONTROL: u32 = 0x3;

const DEFAULT_WINDOW: i64 = 65_535;
const MAX_WINDOW: i64 = 0x7fff_ffff;
const MAX_DATA_LEN: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlowScope {
    Connection,
    Stream(u32),
}

#[derive(Debug, PartialEq, Eq)]
struct FlowError {
    scope: FlowScope,
    code: u32,
    message: String,
}

impl FlowError {
    fn connection(code: u32, message: impl Into<String>) -> Self {
        Self {
            scope: FlowScope::Connection,
            code,
            message: message.into(),
        }
    }

    fn stream(stream_id: u32, code: u32, message: impl Into<String>) -> Self {
        Self {
            scope: FlowScope::Stream(stream_id),
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for FlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FlowError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SendState {
    Open,
    Reset,
    GoneAway,
}

#[derive(Debug)]
struct FlowControl {
    connection: i64,
    initial_stream: i64,
    streams: HashMap<u32, i64>,
}

impl FlowControl {
    fn new() -> Self {
        Self {
            connection: DEFAULT_WINDOW,
            initial_stream: DEFAULT_WINDOW,
            streams: HashMap::new(),
        }
    }

    fn open_stream(&mut self, stream_id: u32) {
        self.streams.entry(stream_id).or_insert(self.initial_stream);
    }

    fn close_stream(&mut self, stream_id: u32) {
        self.streams.remove(&stream_id);
    }

    fn has_stream(&self, stream_id: u32) -> bool {
        self.streams.contains_key(&stream_id)
    }

    fn stream_count(&self) -> usize {
        self.streams.len()
    }

    fn available(&self, stream_id: u32) -> i64 {
        let stream = self.streams.get(&stream_id).copied().unwrap_or(0);
        stream.min(self.connection).max(0)
    }

    fn consume(&mut self, stream_id: u32, len: usize) -> Result<(), FlowError> {
        let len = i64::try_from(len)
            .map_err(|_| FlowError::connection(ERR_FLOW_CONTROL, "data length overflow"))?;
        let limit = if stream_id == 0 {
            self.connection
        } else {
            self.available(stream_id)
        };
        if len > limit || len > self.connection {
            return Err(FlowError::connection(
                ERR_FLOW_CONTROL,
                "data exceeds available window",
            ));
        }
        self.connection -= len;
        if let Some(window) = self.streams.get_mut(&stream_id) {
            *window -= len;
        }
        Ok(())
    }

    fn apply_initial_window(&mut self, value: u32) -> Result<(), FlowError> {
        let value = i64::from(value);
        if value > MAX_WINDOW {
            return Err(FlowError::connection(
                ERR_FLOW_CONTROL,
                "initial window size above maximum",
            ));
        }
        let delta = value - self.initial_stream;
        let mut next = Vec::with_capacity(self.streams.len());
        for (stream_id, window) in &self.streams {
            let updated = *window + delta;
            if updated > MAX_WINDOW {
                return Err(FlowError::connection(
                    ERR_FLOW_CONTROL,
                    "stream window overflow after settings change",
                ));
            }
            next.push((*stream_id, updated));
        }
        for (stream_id, window) in next {
            if let Some(slot) = self.streams.get_mut(&stream_id) {
                *slot = window;
            }
        }
        self.initial_stream = value;
        Ok(())
    }

    fn window_update(&mut self, stream_id: u32, increment: u32) -> Result<(), FlowError> {
        if increment == 0 {
            return Err(if stream_id == 0 {
                FlowError::connection(ERR_PROTOCOL, "zero connection window increment")
            } else {
                FlowError::stream(stream_id, ERR_PROTOCOL, "zero stream window increment")
            });
        }
        let increment = i64::from(increment);
        if stream_id == 0 {
            let updated = self.connection + increment;
            if updated > MAX_WINDOW {
                return Err(FlowError::connection(
                    ERR_FLOW_CONTROL,
                    "connection window overflow",
                ));
            }
            self.connection = updated;
            return Ok(());
        }
        let Some(window) = self.streams.get_mut(&stream_id) else {
            return Ok(());
        };
        let updated = *window + increment;
        if updated > MAX_WINDOW {
            return Err(FlowError::stream(
                stream_id,
                ERR_FLOW_CONTROL,
                "stream window overflow",
            ));
        }
        *window = updated;
        Ok(())
    }

    #[cfg(test)]
    fn connection_window(&self) -> i64 {
        self.connection
    }

    #[cfg(test)]
    fn stream_window(&self, stream_id: u32) -> Option<i64> {
        self.streams.get(&stream_id).copied()
    }

    #[cfg(test)]
    fn initial_window(&self) -> i64 {
        self.initial_stream
    }
}

pub fn is_preface(buf: &[u8]) -> bool {
    !buf.is_empty() && buf.len() <= PREFACE.len() && PREFACE[..buf.len()] == *buf
}

struct Frame {
    typ: u8,
    flags: u8,
    stream_id: u32,
    payload: Vec<u8>,
    flow_applied: bool,
}

const MAX_DEFERRED_FRAMES: usize = 64;
const MAX_DEFERRED_BYTES: usize = 1 << 20;
const MAX_POLL_FRAMES: usize = 64;

struct Conn {
    stream: TcpStream,
    flow: FlowControl,
    recv_credit: u32,
    peer_gone: bool,
    deferred: std::collections::VecDeque<Frame>,
    deferred_bytes: usize,
    inbuf: Vec<u8>,
    out: Vec<u8>,
    cancel_interval: usize,
}

impl Conn {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            flow: FlowControl::new(),
            recv_credit: 0,
            peer_gone: false,
            deferred: std::collections::VecDeque::new(),
            deferred_bytes: 0,
            inbuf: Vec::with_capacity(16 * 1024),
            out: Vec::with_capacity(16 * 1024 + 9),
            cancel_interval: crate::config::DEFAULT_H2_CANCEL_INTERVAL,
        }
    }

    fn read_preface(&mut self) -> Result<()> {
        let mut buf = [0u8; 24];
        self.stream.read_exact(&mut buf)?;
        if buf != PREFACE {
            return Err(anyhow!("bad http/2 preface"));
        }
        Ok(())
    }

    fn read_into_buf(&mut self) -> std::io::Result<usize> {
        let mut tmp = [0u8; 16 * 1024];
        let n = self.stream.read(&mut tmp)?;
        if n > 0 {
            self.inbuf.extend_from_slice(&tmp[..n]);
        }
        Ok(n)
    }

    fn take_frame(&mut self) -> Result<Option<Frame>> {
        if self.inbuf.len() < 9 {
            return Ok(None);
        }
        let head = &self.inbuf[..9];
        let len = ((head[0] as usize) << 16) | ((head[1] as usize) << 8) | head[2] as usize;
        if len > 1 << 20 {
            return Err(anyhow!("http/2 frame too large: {len}"));
        }
        if self.inbuf.len() < 9 + len {
            return Ok(None);
        }
        let typ = head[3];
        let flags = head[4];
        let stream_id = u32::from_be_bytes([head[5] & 0x7f, head[6], head[7], head[8]]);
        let payload = self.inbuf[9..9 + len].to_vec();
        self.inbuf.drain(..9 + len);
        Ok(Some(Frame {
            typ,
            flags,
            stream_id,
            payload,
            flow_applied: false,
        }))
    }

    fn read_frame(&mut self) -> Result<Frame> {
        loop {
            if let Some(frame) = self.take_frame()? {
                return Ok(frame);
            }
            if self.read_into_buf()? == 0 {
                return Err(anyhow!("http/2 connection closed"));
            }
        }
    }

    fn poll_peer(&mut self, stream_id: u32) -> Result<SendState> {
        self.stream.set_nonblocking(true)?;
        let state = self.poll_peer_frames(stream_id);
        self.stream.set_nonblocking(false)?;
        state
    }

    fn poll_peer_frames(&mut self, stream_id: u32) -> Result<SendState> {
        let mut budget = MAX_POLL_FRAMES;
        loop {
            while let Some(frame) = self.take_frame()? {
                if let Some(state) = self.dispatch_peer_frame(frame, stream_id)? {
                    return Ok(state);
                }
                budget -= 1;
                if budget == 0 {
                    return Ok(SendState::Open);
                }
            }
            match self.read_into_buf() {
                Ok(0) => {
                    self.peer_gone = true;
                    return Ok(SendState::GoneAway);
                }
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::WouldBlock => return Ok(SendState::Open),
                Err(e) if e.kind() == ErrorKind::ConnectionReset => {
                    self.peer_gone = true;
                    return Ok(SendState::GoneAway);
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn dispatch_peer_frame(&mut self, frame: Frame, stream_id: u32) -> Result<Option<SendState>> {
        match frame.typ {
            FRAME_SETTINGS => {
                if frame.flags & FLAG_ACK == 0 {
                    self.apply_settings(&frame.payload)?;
                }
            }
            FRAME_PING => {
                if frame.flags & FLAG_ACK == 0 && frame.payload.len() == 8 {
                    self.ping_ack(&frame.payload)?;
                }
            }
            FRAME_WINDOW_UPDATE => self.apply_window_update(&frame)?,
            FRAME_RST => {
                self.flow.close_stream(frame.stream_id);
                if frame.stream_id == stream_id {
                    return Ok(Some(SendState::Reset));
                }
            }
            FRAME_GOAWAY => {
                self.peer_gone = true;
                return Ok(Some(SendState::GoneAway));
            }
            FRAME_HEADERS | FRAME_CONTINUATION => {
                self.flow.open_stream(frame.stream_id);
                self.defer(frame)?;
            }
            FRAME_DATA => {
                let len = frame.payload.len() as u32;
                if len > 0 {
                    self.window_update(0, len)?;
                    if self.flow.has_stream(frame.stream_id) {
                        self.window_update(frame.stream_id, len)?;
                    }
                }
                let mut frame = frame;
                frame.flow_applied = true;
                self.defer(frame)?;
            }
            _ => {}
        }
        Ok(None)
    }

    fn defer(&mut self, frame: Frame) -> Result<()> {
        self.deferred_bytes += frame.payload.len();
        if self.deferred.len() >= MAX_DEFERRED_FRAMES || self.deferred_bytes > MAX_DEFERRED_BYTES {
            return Err(anyhow!("too many pipelined http/2 frames"));
        }
        self.deferred.push_back(frame);
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Frame> {
        if let Some(frame) = self.deferred.pop_front() {
            self.deferred_bytes = self.deferred_bytes.saturating_sub(frame.payload.len());
            Ok(frame)
        } else {
            self.read_frame()
        }
    }

    fn write_frame(&mut self, typ: u8, flags: u8, stream_id: u32, payload: &[u8]) -> Result<()> {
        let len = payload.len();
        if len > 0x00ff_ffff {
            return Err(anyhow!("http/2 payload too large"));
        }
        let mut head = [0u8; 9];
        head[0] = (len >> 16) as u8;
        head[1] = (len >> 8) as u8;
        head[2] = len as u8;
        head[3] = typ;
        head[4] = flags;
        head[5..9].copy_from_slice(&(stream_id & 0x7fff_ffff).to_be_bytes());
        let Self { stream, out, .. } = self;
        out.clear();
        out.extend_from_slice(&head);
        out.extend_from_slice(payload);
        stream.write_all(out)?;
        stream.flush()?;
        Ok(())
    }

    fn send_settings(&mut self) -> Result<()> {
        let mut payload = Vec::with_capacity(20);
        for (id, value) in [(0x1u16, 4096u32), (0x3, 8), (0x4, 1 << 20), (0x5, 16_384)] {
            payload.extend_from_slice(&id.to_be_bytes());
            payload.extend_from_slice(&value.to_be_bytes());
        }
        self.write_frame(FRAME_SETTINGS, 0, 0, &payload)
    }

    fn settings_ack(&mut self) -> Result<()> {
        self.write_frame(FRAME_SETTINGS, FLAG_ACK, 0, &[])
    }

    fn ping_ack(&mut self, payload: &[u8]) -> Result<()> {
        self.write_frame(FRAME_PING, FLAG_ACK, 0, payload)
    }

    fn window_update(&mut self, stream_id: u32, inc: u32) -> Result<()> {
        self.write_frame(FRAME_WINDOW_UPDATE, 0, stream_id, &inc.to_be_bytes())
    }

    fn send_goaway(&mut self, last_stream_id: u32, code: u32) -> Result<()> {
        let mut payload = Vec::with_capacity(8);
        payload.extend_from_slice(&(last_stream_id & 0x7fff_ffff).to_be_bytes());
        payload.extend_from_slice(&code.to_be_bytes());
        self.write_frame(FRAME_GOAWAY, 0, 0, &payload)
    }

    fn send_rst(&mut self, stream_id: u32, code: u32) -> Result<()> {
        self.write_frame(FRAME_RST, 0, stream_id, &code.to_be_bytes())
    }

    fn handle_flow_error(&mut self, error: FlowError) -> Result<()> {
        match error.scope {
            FlowScope::Connection => {
                let _ = self.send_goaway(0, error.code);
                Err(anyhow!(error.message))
            }
            FlowScope::Stream(stream_id) => {
                let _ = self.send_rst(stream_id, error.code);
                self.flow.close_stream(stream_id);
                Ok(())
            }
        }
    }

    fn apply_settings(&mut self, payload: &[u8]) -> Result<()> {
        if !payload.len().is_multiple_of(6) {
            return Err(anyhow!("bad settings frame"));
        }
        for pair in payload.as_chunks::<6>().0 {
            let id = u16::from_be_bytes([pair[0], pair[1]]);
            let value = u32::from_be_bytes([pair[2], pair[3], pair[4], pair[5]]);
            if id == SETTINGS_INITIAL_WINDOW_SIZE
                && let Err(error) = self.flow.apply_initial_window(value)
            {
                self.handle_flow_error(error)?;
            }
        }
        self.settings_ack()
    }

    fn apply_window_update(&mut self, frame: &Frame) -> Result<()> {
        if frame.payload.len() != 4 {
            return Err(anyhow!("bad window update frame"));
        }
        let increment = u32::from_be_bytes([
            frame.payload[0],
            frame.payload[1],
            frame.payload[2],
            frame.payload[3],
        ]) & 0x7fff_ffff;
        if let Err(error) = self.flow.window_update(frame.stream_id, increment) {
            self.handle_flow_error(error)?;
        }
        Ok(())
    }

    fn send_headers(&mut self, stream_id: u32, block: &[u8], end_stream: bool) -> Result<()> {
        let mut flags = FLAG_END_HEADERS;
        if end_stream {
            flags |= FLAG_END_STREAM;
        }
        self.write_frame(FRAME_HEADERS, flags, stream_id, block)
    }

    fn send_data(&mut self, stream_id: u32, payload: &[u8], end_stream: bool) -> Result<SendState> {
        if payload.is_empty() {
            if end_stream {
                self.write_frame(FRAME_DATA, FLAG_END_STREAM, stream_id, &[])?;
                self.flow.close_stream(stream_id);
            }
            return Ok(SendState::Open);
        }
        self.flow.open_stream(stream_id);
        let mut offset = 0usize;
        while offset < payload.len() {
            let available = self.flow.available(stream_id);
            if available == 0 {
                match self.wait_for_window(stream_id)? {
                    SendState::Open => continue,
                    state => return Ok(state),
                }
            }
            let len = usize::try_from(available)
                .unwrap_or(usize::MAX)
                .min(payload.len() - offset)
                .min(MAX_DATA_LEN);
            let last = offset + len == payload.len();
            let flags = if end_stream && last {
                FLAG_END_STREAM
            } else {
                0
            };
            self.write_frame(FRAME_DATA, flags, stream_id, &payload[offset..offset + len])?;
            if let Err(error) = self.flow.consume(stream_id, len) {
                return match self.handle_flow_error(error) {
                    Ok(()) => Ok(SendState::Reset),
                    Err(error) => Err(error),
                };
            }
            offset += len;
        }
        if end_stream {
            self.flow.close_stream(stream_id);
        }
        Ok(SendState::Open)
    }

    fn wait_for_window(&mut self, stream_id: u32) -> Result<SendState> {
        loop {
            if !self.flow.has_stream(stream_id) {
                return Ok(SendState::Reset);
            }
            let frame = self.read_frame()?;
            if let Some(state) = self.dispatch_peer_frame(frame, stream_id)? {
                return Ok(state);
            }
            if self.flow.available(stream_id) > 0 {
                return Ok(SendState::Open);
            }
        }
    }
}

const RECV_WINDOW_HALF: u32 = 32 * 1024;
const MAX_CONCURRENT_STREAMS: usize = 8;

struct Pending {
    id: u32,
    method: String,
    path: String,
    body: Vec<u8>,
    end_stream: bool,
}

pub fn serve(
    stream: TcpStream,
    engine: &Engine,
    info: &crate::serve::ServerInfo,
    limits: Limits,
) -> Result<()> {
    let max_body = limits.max_body;
    let mut conn = Conn::new(stream);
    conn.cancel_interval = limits.h2_cancel_interval;
    conn.read_preface()?;
    conn.send_settings()?;
    let mut ctx = HpackCtx::new();
    let mut block: Vec<u8> = Vec::new();
    let mut pending: Option<Pending> = None;
    loop {
        if conn.peer_gone {
            break;
        }
        let frame = conn.next_frame()?;
        match frame.typ {
            FRAME_SETTINGS => {
                if frame.flags & FLAG_ACK == 0 {
                    if frame.payload.len() % 6 != 0 {
                        return Err(anyhow!("bad settings frame"));
                    }
                    for pair in frame.payload.as_chunks::<6>().0 {
                        let id = u16::from_be_bytes([pair[0], pair[1]]);
                        let value = u32::from_be_bytes([pair[2], pair[3], pair[4], pair[5]]);
                        if id == SETTINGS_HEADER_TABLE_SIZE {
                            let capped = (value as usize).min(hpack::HPACK_MAX_TABLE);
                            ctx.set_max_size(capped)?;
                        }
                    }
                    conn.apply_settings(&frame.payload)?;
                }
            }
            FRAME_PING => {
                if frame.flags & FLAG_ACK == 0 && frame.payload.len() == 8 {
                    conn.ping_ack(&frame.payload)?;
                }
            }
            FRAME_HEADERS => {
                let mut off = 0usize;
                let mut pad = 0usize;
                if !conn.flow.has_stream(frame.stream_id)
                    && conn.flow.stream_count() >= MAX_CONCURRENT_STREAMS
                {
                    conn.send_rst(frame.stream_id, 7)?;
                    continue;
                }
                conn.flow.open_stream(frame.stream_id);
                if frame.flags & FLAG_PADDED != 0 {
                    pad = *frame
                        .payload
                        .first()
                        .ok_or_else(|| anyhow!("padded headers"))?
                        as usize;
                    off = 1;
                }
                if frame.flags & FLAG_PRIORITY != 0 {
                    off += 5;
                }
                let end = frame.payload.len().saturating_sub(pad);
                if off > end {
                    return Err(anyhow!("bad headers frame"));
                }
                block = frame.payload[off..end].to_vec();
                let end_stream = frame.flags & FLAG_END_STREAM != 0;
                if frame.flags & FLAG_END_HEADERS != 0 {
                    pending = Some(build_request(
                        &block,
                        frame.stream_id,
                        end_stream,
                        &mut ctx,
                    )?);
                    if end_stream {
                        let req = pending.take().ok_or_else(|| anyhow!("missing request"))?;
                        handle(&mut conn, engine, info, req)?;
                    }
                } else {
                    pending = Some(Pending {
                        id: frame.stream_id,
                        method: String::new(),
                        path: String::new(),
                        body: Vec::new(),
                        end_stream,
                    });
                }
            }
            FRAME_CONTINUATION => {
                if block.len() + frame.payload.len() > 1 << 20 {
                    return Err(anyhow!("header block too large"));
                }
                block.extend_from_slice(&frame.payload);
                if frame.flags & FLAG_END_HEADERS != 0 {
                    let stream_id = pending.as_ref().map(|p| p.id).unwrap_or(frame.stream_id);
                    let end_stream = pending.as_ref().map(|p| p.end_stream).unwrap_or(false);
                    pending = Some(build_request(&block, stream_id, end_stream, &mut ctx)?);
                    if end_stream {
                        let req = pending.take().ok_or_else(|| anyhow!("missing request"))?;
                        handle(&mut conn, engine, info, req)?;
                    }
                }
            }
            FRAME_DATA => {
                if let Some(req) = pending.as_mut() {
                    if req.id != frame.stream_id {
                        conn.send_rst(frame.stream_id, 2)?;
                        continue;
                    }
                    if req.body.len() + frame.payload.len() > max_body {
                        conn.send_rst(frame.stream_id, 3)?;
                        pending = None;
                        continue;
                    }
                    req.body.extend_from_slice(&frame.payload);
                    if !frame.flow_applied && !frame.payload.is_empty() {
                        let inc = frame.payload.len() as u32;
                        conn.recv_credit += inc;
                        if conn.recv_credit >= RECV_WINDOW_HALF {
                            let credit = conn.recv_credit;
                            conn.recv_credit = 0;
                            conn.window_update(0, credit)?;
                            if conn.flow.has_stream(frame.stream_id) {
                                conn.window_update(frame.stream_id, credit)?;
                            }
                        }
                    }
                    if frame.flags & FLAG_END_STREAM != 0 {
                        let req = pending.take().ok_or_else(|| anyhow!("missing request"))?;
                        handle(&mut conn, engine, info, req)?;
                    }
                }
            }
            FRAME_RST => {
                conn.flow.close_stream(frame.stream_id);
                pending = None;
            }
            FRAME_GOAWAY => break,
            FRAME_WINDOW_UPDATE => conn.apply_window_update(&frame)?,
            _ => {}
        }
    }
    Ok(())
}

fn build_request(
    block: &[u8],
    stream_id: u32,
    end_stream: bool,
    ctx: &mut HpackCtx,
) -> Result<Pending> {
    let headers = hpack::decode(block, ctx)?;
    let mut method = String::new();
    let mut path = String::new();
    for (name, value) in headers {
        match name.as_str() {
            ":method" => method = value,
            ":path" => path = value,
            _ => {}
        }
    }
    if method.is_empty() || path.is_empty() {
        return Err(anyhow!("missing :method or :path"));
    }
    Ok(Pending {
        id: stream_id,
        method,
        path,
        body: Vec::new(),
        end_stream,
    })
}

fn send_json(conn: &mut Conn, stream_id: u32, status: &str, body: &str) -> Result<()> {
    let block = hpack::encode(&[
        (":status", status.to_string()),
        ("content-type", "application/json".to_string()),
        ("content-length", body.len().to_string()),
    ]);
    conn.send_headers(stream_id, &block, false)?;
    conn.send_data(stream_id, body.as_bytes(), true)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Health,
    Models,
    Chat,
    Completions,
    Metrics,
    NotFound,
}

fn route(method: &str, path: &str) -> Route {
    match (method, path) {
        ("GET", "/health") => Route::Health,
        ("GET", "/v1/models") => Route::Models,
        ("POST", "/v1/chat/completions") => Route::Chat,
        ("POST", "/v1/completions") => Route::Completions,
        ("GET", "/metrics") => Route::Metrics,
        _ => Route::NotFound,
    }
}

fn handle(conn: &mut Conn, engine: &Engine, info: &crate::serve::ServerInfo, req: Pending) -> Result<()> {
    match route(&req.method, &req.path) {
        Route::Health => {
            let code = if engine.alive() { "200" } else { "503" };
            send_json(conn, req.id, code, &info.health_json())
        }
        Route::Models => send_json(conn, req.id, "200", &models_json(&info.model, 0)),
        Route::Chat => chat(conn, engine, &info.model, req),
        Route::Completions => completions(conn, engine, &info.model, req),
        Route::Metrics => send_json(
            conn,
            req.id,
            "200",
            &metrics_json(&info.model, &engine.metrics()),
        ),
        Route::NotFound => send_json(
            conn,
            req.id,
            "404",
            &json!({"error": {"message": "not found"}}).to_string(),
        ),
    }
}

fn chat(conn: &mut Conn, engine: &Engine, model: &str, req: Pending) -> Result<()> {
    let parsed: ChatRequest = match serde_json::from_slice(&req.body) {
        Ok(parsed) => parsed,
        Err(e) => {
            return send_json(
                conn,
                req.id,
                "400",
                &json!({"error": {"message": format!("bad json: {e}")}}).to_string(),
            );
        }
    };
    let messages = match parsed.to_messages() {
        Ok(messages) => messages,
        Err(e) => {
            return send_json(
                conn,
                req.id,
                "400",
                &json!({"error": {"message": format!("{e}")}}).to_string(),
            );
        }
    };
    let params = parsed.to_params();
    let job = ChatJob::chat(messages, params, parsed.thinking(), parsed.to_tools());
    let _cancel = crate::serve::engine::CancelGuard::new(job.cancel_handle());
    let events = engine.submit(job)?;
    let name = parsed.model.as_deref().unwrap_or(model);
    let meta = CompletionMeta::new(name);
    if parsed.stream {
        let block = hpack::encode(&[
            (":status", "200".to_string()),
            ("content-type", "text/event-stream".to_string()),
            ("cache-control", "no-cache".to_string()),
        ]);
        conn.send_headers(req.id, &block, false)?;
        let mut tool_index = 0usize;
        let mut finish_reason = "stop";
        let mut completed = false;
        let mut rounds = 0usize;
        let mut batch: Vec<u8> = Vec::with_capacity(8192);
        pump_events(&events, |events| {
            batch.clear();
            for event in events {
                match event {
                    Event::Delta(text) => push_sse(&mut batch, |b| {
                        meta.write_chunk(b, Delta::content(text), None)
                    })?,
                    Event::Thinking(text) => push_sse(&mut batch, |b| {
                        meta.write_chunk(b, Delta::reasoning(text), None)
                    })?,
                    Event::ToolCall(tc) => {
                        push_sse(&mut batch, |b| {
                            meta.write_chunk(b, Delta::tool_call(tool_index, tc), None)
                        })?;
                        tool_index += 1;
                    }
                    Event::Final {
                        finish_reason: reason,
                        ..
                    } => {
                        finish_reason = reason;
                        completed = true;
                    }
                    Event::Error(e) => {
                        let text = format!("\n[error] {e}");
                        push_sse(&mut batch, |b| {
                            meta.write_chunk(b, Delta::content(&text), None)
                        })?;
                    }
                }
            }
            if batch.is_empty() {
                return Ok(Pump::Continue);
            }
            if conn.send_data(req.id, &batch, false)? != SendState::Open {
                return Ok(Pump::Stop);
            }
            rounds += 1;
            if rounds >= conn.cancel_interval {
                rounds = 0;
                if conn.poll_peer(req.id)? != SendState::Open {
                    return Ok(Pump::Stop);
                }
            }
            Ok(Pump::Continue)
        })?;
        if !completed {
            return Ok(());
        }
        let mut tail = Vec::with_capacity(256);
        push_sse(&mut tail, |b| {
            meta.write_chunk(b, Delta::empty(), Some(finish_reason))
        })?;
        push_sse_data(&mut tail, b"[DONE]");
        conn.send_data(req.id, &tail, true)?;
        return Ok(());
    }
    let mut text = String::new();
    let mut thinking: Option<String> = None;
    let mut tool_calls = Vec::new();
    let mut finish_reason = "stop";
    let mut usage = (0usize, 0usize);
    for event in events {
        match event {
            Event::Final {
                text: t,
                thinking: th,
                tool_calls: calls,
                finish_reason: reason,
                prompt_tokens,
                completion_tokens,
            } => {
                text = t;
                thinking = th;
                tool_calls = calls;
                finish_reason = reason;
                usage = (prompt_tokens, completion_tokens);
            }
            Event::Error(e) => {
                return send_json(
                    conn,
                    req.id,
                    "500",
                    &json!({"error": {"message": e}}).to_string(),
                );
            }
            Event::Delta(_) | Event::Thinking(_) | Event::ToolCall(_) => {}
        }
    }
    send_json(
        conn,
        req.id,
        "200",
        &meta.message_response(
            &text,
            thinking.as_deref(),
            &tool_calls,
            finish_reason,
            usage.0,
            usage.1,
        ),
    )
}

fn completions(conn: &mut Conn, engine: &Engine, model: &str, req: Pending) -> Result<()> {
    let parsed: CompletionRequest = match serde_json::from_slice(&req.body) {
        Ok(parsed) => parsed,
        Err(e) => {
            return send_json(
                conn,
                req.id,
                "400",
                &json!({"error": {"message": format!("bad json: {e}")}}).to_string(),
            );
        }
    };
    let prompts = match parsed.to_prompts() {
        Ok(prompts) => prompts,
        Err(e) => {
            return send_json(
                conn,
                req.id,
                "400",
                &json!({"error": {"message": format!("{e}")}}).to_string(),
            );
        }
    };
    let params = parsed.to_params();
    let name = parsed.model.as_deref().unwrap_or(model);
    let meta = CompletionMeta::new_completion(name);
    if parsed.stream {
        if prompts.len() != 1 {
            return send_json(
                conn,
                req.id,
                "400",
                &json!({"error": {"message": "streaming requires a single prompt"}}).to_string(),
            );
        }
        let job = ChatJob::completion(prompts[0].clone(), params);
        let _cancel = crate::serve::engine::CancelGuard::new(job.cancel_handle());
        let events = engine.submit(job)?;
        let block = hpack::encode(&[
            (":status", "200".to_string()),
            ("content-type", "text/event-stream".to_string()),
            ("cache-control", "no-cache".to_string()),
        ]);
        conn.send_headers(req.id, &block, false)?;
        let mut batch: Vec<u8> = Vec::with_capacity(8192);
        push_sse(&mut batch, |b| meta.write_completion_chunk(b, "", None))?;
        conn.send_data(req.id, &batch, false)?;
        let mut finish_reason = "stop";
        let mut completed = false;
        let mut rounds = 0usize;
        pump_events(&events, |events| {
            batch.clear();
            for event in events {
                match event {
                    Event::Delta(text) => {
                        push_sse(&mut batch, |b| meta.write_completion_chunk(b, text, None))?;
                    }
                    Event::Final {
                        finish_reason: reason,
                        ..
                    } => {
                        finish_reason = reason;
                        completed = true;
                    }
                    Event::Error(e) => {
                        let text = format!("\n[error] {e}");
                        push_sse(&mut batch, |b| meta.write_completion_chunk(b, &text, None))?;
                    }
                    Event::Thinking(_) | Event::ToolCall(_) => {}
                }
            }
            if batch.is_empty() {
                return Ok(Pump::Continue);
            }
            if conn.send_data(req.id, &batch, false)? != SendState::Open {
                return Ok(Pump::Stop);
            }
            rounds += 1;
            if rounds >= conn.cancel_interval {
                rounds = 0;
                if conn.poll_peer(req.id)? != SendState::Open {
                    return Ok(Pump::Stop);
                }
            }
            Ok(Pump::Continue)
        })?;
        if !completed {
            return Ok(());
        }
        let mut tail = Vec::with_capacity(256);
        push_sse(&mut tail, |b| {
            meta.write_completion_chunk(b, "", Some(finish_reason))
        })?;
        push_sse_data(&mut tail, b"[DONE]");
        conn.send_data(req.id, &tail, true)?;
        return Ok(());
    }
    let mut choices = Vec::with_capacity(prompts.len());
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    for prompt in prompts {
        let job = ChatJob::completion(prompt, params.clone());
        let _cancel = crate::serve::engine::CancelGuard::new(job.cancel_handle());
        let events = engine.submit(job)?;
        let mut text = String::new();
        let mut finish_reason = "stop";
        for event in events {
            match event {
                Event::Final {
                    text: t,
                    finish_reason: reason,
                    prompt_tokens: p,
                    completion_tokens: c,
                    ..
                } => {
                    text = t;
                    finish_reason = reason;
                    prompt_tokens += p;
                    completion_tokens += c;
                }
                Event::Error(e) => {
                    return send_json(
                        conn,
                        req.id,
                        "500",
                        &json!({"error": {"message": e}}).to_string(),
                    );
                }
                Event::Delta(_) | Event::Thinking(_) | Event::ToolCall(_) => {}
            }
        }
        choices.push((text, finish_reason.to_string()));
    }
    send_json(
        conn,
        req.id,
        "200",
        &meta.completion_response(&choices, prompt_tokens, completion_tokens),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::time::Duration;

    fn pair() -> (Conn, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        (Conn::new(server), client)
    }

    fn read_frame(stream: &mut TcpStream) -> (u8, u8, u32, Vec<u8>) {
        let mut head = [0u8; 9];
        stream.read_exact(&mut head).unwrap();
        let len = ((head[0] as usize) << 16) | ((head[1] as usize) << 8) | head[2] as usize;
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).unwrap();
        let stream_id = u32::from_be_bytes([head[5] & 0x7f, head[6], head[7], head[8]]);
        (head[3], head[4], stream_id, payload)
    }

    fn send_frame(stream: &mut TcpStream, typ: u8, flags: u8, stream_id: u32, payload: &[u8]) {
        let len = payload.len();
        let mut head = [0u8; 9];
        head[0] = (len >> 16) as u8;
        head[1] = (len >> 8) as u8;
        head[2] = len as u8;
        head[3] = typ;
        head[4] = flags;
        head[5..9].copy_from_slice(&(stream_id & 0x7fff_ffff).to_be_bytes());
        stream.write_all(&head).unwrap();
        stream.write_all(payload).unwrap();
        stream.flush().unwrap();
    }

    #[test]
    fn frame_layout_roundtrip() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            stream.write_all(PREFACE).unwrap();
            let mut head = [0u8; 9];
            stream.read_exact(&mut head).unwrap();
            let len = ((head[0] as usize) << 16) | ((head[1] as usize) << 8) | head[2] as usize;
            let mut payload = vec![0u8; len];
            stream.read_exact(&mut payload).unwrap();
            (
                head[3],
                u32::from_be_bytes([head[5], head[6], head[7], head[8]]),
                payload,
            )
        });
        let (stream, _) = listener.accept().unwrap();
        let mut conn = Conn::new(stream);
        conn.read_preface().unwrap();
        conn.send_settings().unwrap();
        let (typ, sid, payload) = client.join().unwrap();
        assert_eq!(typ, FRAME_SETTINGS);
        assert_eq!(sid, 0);
        assert_eq!(payload.len(), 24);
    }

    #[test]
    fn detects_preface_prefixes() {
        assert!(is_preface(b"PRI * HTTP/2.0\r\n"));
        assert!(is_preface(PREFACE));
        assert!(!is_preface(b"GET / HTTP/1.1"));
        assert!(!is_preface(b""));
    }

    #[test]
    fn windows_start_from_default_initial_size() {
        let mut flow = FlowControl::new();
        assert_eq!(flow.initial_window(), DEFAULT_WINDOW);
        assert_eq!(flow.connection_window(), DEFAULT_WINDOW);
        flow.open_stream(1);
        assert_eq!(flow.stream_window(1), Some(DEFAULT_WINDOW));
        assert_eq!(flow.available(1), DEFAULT_WINDOW);
        flow.close_stream(1);
        assert!(!flow.has_stream(1));
        assert_eq!(flow.available(1), 0);
    }

    #[test]
    fn initial_window_from_settings_applies_to_new_streams() {
        let mut flow = FlowControl::new();
        flow.apply_initial_window(4096).unwrap();
        flow.open_stream(3);
        assert_eq!(flow.stream_window(3), Some(4096));
        assert_eq!(flow.available(3), 4096);
    }

    #[test]
    fn consume_reduces_stream_and_connection_windows() {
        let mut flow = FlowControl::new();
        flow.open_stream(1);
        flow.consume(1, 5000).unwrap();
        assert_eq!(flow.stream_window(1), Some(DEFAULT_WINDOW - 5000));
        assert_eq!(flow.connection_window(), DEFAULT_WINDOW - 5000);
        assert_eq!(flow.available(1), DEFAULT_WINDOW - 5000);
    }

    #[test]
    fn connection_window_limits_available() {
        let mut flow = FlowControl::new();
        flow.open_stream(1);
        flow.consume(0, 65_000).unwrap();
        assert_eq!(flow.stream_window(1), Some(DEFAULT_WINDOW));
        assert_eq!(flow.connection_window(), 535);
        assert_eq!(flow.available(1), 535);
        flow.consume(1, 535).unwrap();
        assert_eq!(flow.available(1), 0);
        assert_eq!(flow.stream_window(1), Some(DEFAULT_WINDOW - 535));
    }

    #[test]
    fn window_update_refills_stream_and_connection() {
        let mut flow = FlowControl::new();
        flow.open_stream(1);
        flow.consume(1, DEFAULT_WINDOW as usize).unwrap();
        assert_eq!(flow.available(1), 0);
        flow.window_update(1, 1000).unwrap();
        assert_eq!(flow.stream_window(1), Some(1000));
        assert_eq!(flow.available(1), 0);
        flow.window_update(0, 1000).unwrap();
        assert_eq!(flow.connection_window(), 1000);
        assert_eq!(flow.available(1), 1000);
    }

    #[test]
    fn initial_window_change_applies_negative_delta() {
        let mut flow = FlowControl::new();
        flow.apply_initial_window(1024).unwrap();
        flow.open_stream(1);
        flow.consume(1, 400).unwrap();
        assert_eq!(flow.stream_window(1), Some(624));
        flow.apply_initial_window(0).unwrap();
        assert_eq!(flow.stream_window(1), Some(-400));
        assert_eq!(flow.available(1), 0);
        flow.apply_initial_window(2048).unwrap();
        assert_eq!(flow.stream_window(1), Some(1648));
    }

    #[test]
    fn window_overflow_reports_error_without_panic() {
        let mut flow = FlowControl::new();
        flow.open_stream(1);
        assert!(flow.window_update(0, MAX_WINDOW as u32).is_err());
        assert!(flow.window_update(1, MAX_WINDOW as u32).is_err());
        assert_eq!(
            flow.window_update(0, 0),
            Err(FlowError::connection(
                ERR_PROTOCOL,
                "zero connection window increment"
            ))
        );
        assert_eq!(
            flow.window_update(1, 0),
            Err(FlowError::stream(
                1,
                ERR_PROTOCOL,
                "zero stream window increment"
            ))
        );
        assert!(flow.apply_initial_window(0x8000_0000).is_err());
        flow.apply_initial_window(MAX_WINDOW as u32).unwrap();
        assert_eq!(flow.stream_window(1), Some(MAX_WINDOW));
        let error = flow.window_update(1, 1).unwrap_err();
        assert_eq!(error.scope, FlowScope::Stream(1));
        assert_eq!(error.code, ERR_FLOW_CONTROL);
    }

    #[test]
    fn stream_window_overflow_after_settings_change_is_connection_error() {
        let mut flow = FlowControl::new();
        flow.open_stream(1);
        flow.window_update(1, (MAX_WINDOW - DEFAULT_WINDOW) as u32)
            .unwrap();
        assert_eq!(flow.stream_window(1), Some(MAX_WINDOW));
        let error = flow.apply_initial_window(MAX_WINDOW as u32).unwrap_err();
        assert_eq!(error.scope, FlowScope::Connection);
        assert_eq!(error.code, ERR_FLOW_CONTROL);
    }

    #[test]
    fn window_update_for_unknown_stream_is_ignored() {
        let mut flow = FlowControl::new();
        assert_eq!(flow.window_update(9, 1024), Ok(()));
        assert_eq!(flow.stream_window(9), None);
        assert_eq!(flow.connection_window(), DEFAULT_WINDOW);
        flow.open_stream(9);
        flow.close_stream(9);
        assert_eq!(flow.window_update(9, 1024), Ok(()));
        assert_eq!(flow.stream_window(9), None);
    }

    #[test]
    fn settings_frame_updates_initial_window_and_acks() {
        let (mut conn, mut client) = pair();
        conn.apply_settings(&[0x00, 0x02, 0x00, 0x00, 0x10, 0x00])
            .unwrap();
        assert_eq!(conn.flow.initial_window(), 4096);
        conn.flow.open_stream(3);
        assert_eq!(conn.flow.stream_window(3), Some(4096));
        let (typ, flags, sid, payload) = read_frame(&mut client);
        assert_eq!(typ, FRAME_SETTINGS);
        assert_eq!(flags, FLAG_ACK);
        assert_eq!(sid, 0);
        assert!(payload.is_empty());
    }

    #[test]
    fn send_data_waits_for_window_and_continues() {
        let (mut conn, mut client) = pair();
        conn.flow.apply_initial_window(20).unwrap();
        conn.flow.open_stream(1);
        let payload = vec![0x5au8; 100];
        let expected = payload.clone();
        let writer =
            std::thread::spawn(move || -> SendState { conn.send_data(1, &payload, true).unwrap() });
        let mut received = Vec::new();
        let mut pong = false;
        let mut settings_ack = false;
        let mut first = true;
        loop {
            let (typ, flags, sid, payload) = read_frame(&mut client);
            match typ {
                FRAME_DATA => {
                    assert_eq!(sid, 1);
                    assert!(payload.len() <= 20);
                    received.extend_from_slice(&payload);
                    if first {
                        first = false;
                        send_frame(&mut client, FRAME_PING, 0, 0, b"lfm-flow");
                        send_frame(
                            &mut client,
                            FRAME_SETTINGS,
                            0,
                            0,
                            &[0x00, 0x03, 0x00, 0x00, 0x00, 0x64],
                        );
                    }
                    let inc = payload.len() as u32;
                    if flags & FLAG_END_STREAM != 0 {
                        break;
                    }
                    send_frame(&mut client, FRAME_WINDOW_UPDATE, 0, 0, &inc.to_be_bytes());
                    send_frame(&mut client, FRAME_WINDOW_UPDATE, 0, 1, &inc.to_be_bytes());
                }
                FRAME_PING => {
                    assert_eq!(flags, FLAG_ACK);
                    assert_eq!(payload, b"lfm-flow");
                    pong = true;
                }
                FRAME_SETTINGS => {
                    assert_eq!(flags, FLAG_ACK);
                    settings_ack = true;
                }
                other => panic!("unexpected frame type {other}"),
            }
        }
        assert_eq!(writer.join().unwrap(), SendState::Open);
        assert_eq!(received, expected);
        assert!(pong);
        assert!(settings_ack);
    }

    #[test]
    fn rst_stream_interrupts_window_wait() {
        let (mut conn, mut client) = pair();
        conn.flow.apply_initial_window(0).unwrap();
        conn.flow.open_stream(1);
        let writer = std::thread::spawn(move || conn.send_data(1, b"hello", false).unwrap());
        send_frame(&mut client, FRAME_RST, 0, 1, &ERR_PROTOCOL.to_be_bytes());
        assert_eq!(writer.join().unwrap(), SendState::Reset);
    }

    #[test]
    fn window_wait_defers_pipelined_headers_and_body() {
        let (mut conn, mut client) = pair();
        conn.flow.apply_initial_window(0).unwrap();
        conn.flow.open_stream(1);
        let writer = std::thread::spawn(move || {
            let state = conn.send_data(1, &[7u8; 40], true).unwrap();
            (conn, state)
        });
        send_frame(&mut client, FRAME_HEADERS, FLAG_END_HEADERS, 3, b"\x82");
        send_frame(&mut client, FRAME_DATA, 0, 3, b"body");
        send_frame(&mut client, FRAME_WINDOW_UPDATE, 0, 0, &40u32.to_be_bytes());
        send_frame(&mut client, FRAME_WINDOW_UPDATE, 0, 1, &40u32.to_be_bytes());
        let (mut conn, state) = writer.join().unwrap();
        assert_eq!(state, SendState::Open);
        let headers = conn.next_frame().unwrap();
        assert_eq!(headers.typ, FRAME_HEADERS);
        assert_eq!(headers.stream_id, 3);
        assert!(conn.flow.has_stream(3));
        let data = conn.next_frame().unwrap();
        assert_eq!(data.typ, FRAME_DATA);
        assert_eq!(data.stream_id, 3);
        assert_eq!(data.payload, b"body");
        assert!(data.flow_applied);
    }

    #[test]
    fn goaway_interrupts_window_wait() {
        let (mut conn, mut client) = pair();
        conn.flow.apply_initial_window(0).unwrap();
        conn.flow.open_stream(1);
        let writer = std::thread::spawn(move || conn.send_data(1, b"hello", false).unwrap());
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&ERR_FLOW_CONTROL.to_be_bytes());
        send_frame(&mut client, FRAME_GOAWAY, 0, 0, &payload);
        assert_eq!(writer.join().unwrap(), SendState::GoneAway);
    }

    #[test]
    fn closed_peer_interrupts_window_wait_without_panic() {
        let (mut conn, client) = pair();
        conn.flow.apply_initial_window(0).unwrap();
        conn.flow.open_stream(1);
        drop(client);
        assert!(conn.send_data(1, b"hello", false).is_err());
    }

    #[test]
    fn read_frame_reassembles_partial_reads() {
        let (mut conn, client) = pair();
        let sender = std::thread::spawn(move || {
            let mut stream = client;
            let mut head = [0u8; 9];
            head[2] = 8;
            head[3] = FRAME_PING;
            head[5..9].copy_from_slice(&0u32.to_be_bytes());
            stream.write_all(&head).unwrap();
            stream.flush().unwrap();
            std::thread::sleep(Duration::from_millis(20));
            stream.write_all(b"lfm-ping").unwrap();
            stream.flush().unwrap();
        });
        let frame = conn.read_frame().unwrap();
        assert_eq!(frame.typ, FRAME_PING);
        assert_eq!(frame.payload, b"lfm-ping");
        sender.join().unwrap();
    }

    #[test]
    fn poll_peer_detects_rst_stream_without_window_exhaustion() {
        let (mut conn, mut client) = pair();
        conn.flow.open_stream(1);
        assert_eq!(conn.poll_peer(1).unwrap(), SendState::Open);
        send_frame(&mut client, FRAME_RST, 0, 1, &ERR_PROTOCOL.to_be_bytes());
        let state = wait_for_poll(&mut conn, 1);
        assert_eq!(state, SendState::Reset);
        assert!(!conn.flow.has_stream(1));
    }

    #[test]
    fn poll_peer_detects_closed_client() {
        let (mut conn, client) = pair();
        conn.flow.open_stream(1);
        drop(client);
        assert_eq!(wait_for_poll(&mut conn, 1), SendState::GoneAway);
        assert!(conn.peer_gone);
    }

    #[test]
    fn poll_peer_defers_pipelined_headers() {
        let (mut conn, mut client) = pair();
        conn.flow.open_stream(1);
        send_frame(&mut client, FRAME_HEADERS, FLAG_END_HEADERS, 3, b"\x82");
        for _ in 0..200 {
            conn.poll_peer(1).unwrap();
            if !conn.deferred.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(conn.flow.stream_count(), 2);
        let frame = conn.next_frame().unwrap();
        assert_eq!(frame.typ, FRAME_HEADERS);
        assert_eq!(frame.stream_id, 3);
    }

    fn wait_for_poll(conn: &mut Conn, stream_id: u32) -> SendState {
        for _ in 0..200 {
            let state = conn.poll_peer(stream_id).unwrap();
            if state != SendState::Open {
                return state;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        SendState::Open
    }

    #[test]
    fn metrics_is_routed_for_get_only() {
        assert_eq!(route("GET", "/metrics"), Route::Metrics);
        assert_eq!(route("POST", "/metrics"), Route::NotFound);
        assert_eq!(route("GET", "/metrics/"), Route::NotFound);
        assert_eq!(route("GET", "/health"), Route::Health);
        assert_eq!(route("GET", "/v1/models"), Route::Models);
        assert_eq!(route("POST", "/v1/chat/completions"), Route::Chat);
        assert_eq!(route("POST", "/v1/completions"), Route::Completions);
        assert_eq!(route("GET", "/unknown"), Route::NotFound);
    }

    #[test]
    fn metrics_response_uses_json_headers_and_flow_control() {
        use crate::serve::engine::EngineMetricsSnapshot;

        let (mut conn, mut client) = pair();
        conn.flow.apply_initial_window(64).unwrap();
        conn.flow.open_stream(1);
        let snapshot = EngineMetricsSnapshot {
            ctx: 4096,
            active: 2,
            waiting: 1,
            steps: 7,
            avg_step_ms: 12.5,
            avg_gpu_us: 100,
            avg_logits_us: 200,
            avg_sample_us: 30,
            avg_detok_us: 40,
            rows_total: 11,
            rows_logits: 9,
            kv_used_pages: 3,
            kv_free_pages: 4,
            kv_cached_pages: 5,
            cache_hits: 6,
            reused_tokens: 8,
            uptime_s: 9.25,
        };
        let body = metrics_json("lfm3", &snapshot);
        assert!(body.len() > 64);
        let expected = body.clone();
        let writer = std::thread::spawn(move || {
            send_json(&mut conn, 1, "200", &body).unwrap();
        });

        let (typ, flags, sid, payload) = read_frame(&mut client);
        assert_eq!(typ, FRAME_HEADERS);
        assert_eq!(sid, 1);
        assert_ne!(flags & FLAG_END_HEADERS, 0);
        let headers = hpack::decode(&payload, &mut HpackCtx::new()).unwrap();
        let find = |name: &str| {
            headers
                .iter()
                .find(|(header, _)| header == name)
                .map(|(_, value)| value.clone())
        };
        assert_eq!(find(":status"), Some("200".to_string()));
        assert_eq!(find("content-type"), Some("application/json".to_string()));
        assert_eq!(find("content-length"), Some(expected.len().to_string()));

        let mut received = Vec::new();
        loop {
            let (typ, flags, sid, payload) = read_frame(&mut client);
            assert_eq!(typ, FRAME_DATA);
            assert_eq!(sid, 1);
            assert!(payload.len() <= 64);
            assert!(!payload.is_empty());
            received.extend_from_slice(&payload);
            if flags & FLAG_END_STREAM != 0 {
                break;
            }
            let inc = payload.len() as u32;
            send_frame(&mut client, FRAME_WINDOW_UPDATE, 0, 0, &inc.to_be_bytes());
            send_frame(&mut client, FRAME_WINDOW_UPDATE, 0, 1, &inc.to_be_bytes());
        }
        writer.join().unwrap();
        assert_eq!(received, expected.as_bytes());

        let value: serde_json::Value = serde_json::from_slice(&received).unwrap();
        for key in ["model", "ctx", "active", "steps", "uptime_s"] {
            assert!(value.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(value["model"], "lfm3");
        assert_eq!(value["ctx"], 4096);
        assert_eq!(value["active"], 2);
        assert_eq!(value["steps"], 7);
        assert_eq!(value["uptime_s"].as_f64(), Some(9.25));
    }
}
