mod domain_meta;

use std::collections::BTreeMap;
use std::fs::File;

use memmap2::Mmap;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum GgufError {
    #[error("bad magic")]
    BadMagic,
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("unexpected eof")]
    UnexpectedEof,
    #[error("bad type: {0}")]
    BadType(u32),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("size overflow")]
    SizeOverflow,
    #[error("bad utf8")]
    BadUtf8,
    #[error("nested array")]
    NestedArray,
}

pub type Error = GgufError;

#[repr(u32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            x => Self::Unknown(x),
        }
    }

    pub fn to_u32(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::IQ2_XXS => 16,
            Self::IQ2_XS => 17,
            Self::IQ3_XXS => 18,
            Self::IQ1_S => 19,
            Self::IQ4_NL => 20,
            Self::IQ3_S => 21,
            Self::IQ2_S => 22,
            Self::IQ4_XS => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::IQ1_M => 29,
            Self::BF16 => 30,
            Self::Unknown(x) => *x,
        }
    }

    pub fn type_size(&self) -> Option<(u64, u64)> {
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            Self::BF16 => Some((1, 2)),
            Self::I8 => Some((1, 1)),
            Self::I16 => Some((1, 2)),
            Self::I32 => Some((1, 4)),
            Self::I64 => Some((1, 8)),
            Self::F64 => Some((1, 8)),
            Self::Q4_0 => Some((32, 18)),
            Self::Q4_1 => Some((32, 20)),
            Self::Q5_0 => Some((32, 22)),
            Self::Q5_1 => Some((32, 24)),
            Self::Q8_0 => Some((32, 34)),
            Self::Q8_1 => Some((32, 36)),
            Self::Q2_K => Some((256, 84)),
            Self::Q3_K => Some((256, 110)),
            Self::Q4_K => Some((256, 144)),
            Self::Q5_K => Some((256, 176)),
            Self::Q6_K => Some((256, 210)),
            Self::Q8_K => Some((256, 292)),
            Self::IQ4_NL => Some((32, 18)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(ArrayValue),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayValue {
    pub elem_type: u32,
    pub values: Vec<MetadataValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
    pub byte_size: u64,
}

pub struct GgufFile {
    mmap: Mmap,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
}

impl GgufFile {
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let buf = &mmap[..];
        let (metadata, tensors, data_offset) = parse(buf)?;
        Ok(Self {
            mmap,
            metadata,
            tensors,
            data_offset,
        })
    }

    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_bytes(&self, name: &str) -> anyhow::Result<&[u8]> {
        let ti = self.tensor(name).ok_or_else(|| GgufError::MissingTensor(name.to_string()))?;
        let start = self.data_offset.checked_add(ti.offset).ok_or(GgufError::SizeOverflow)?;
        let end = start.checked_add(ti.byte_size).ok_or(GgufError::SizeOverflow)?;
        if end > self.mmap.len() as u64 {
            return Err(GgufError::UnexpectedEof.into());
        }
        Ok(&self.mmap[start as usize..end as usize])
    }

    pub fn raw(&self) -> &[u8] {
        &self.mmap[..]
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(MetadataValue::U32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(a)) if a.elem_type == 5 => {
                let mut out = Vec::with_capacity(a.values.len());
                for v in &a.values {
                    match v {
                        MetadataValue::I32(x) => out.push(*x),
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn get_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(a)) if a.elem_type == 4 => {
                let mut out = Vec::with_capacity(a.values.len());
                for v in &a.values {
                    match v {
                        MetadataValue::U32(x) => out.push(*x),
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

fn parse(buf: &[u8]) -> anyhow::Result<(BTreeMap<String, MetadataValue>, Vec<TensorInfo>, u64)> {
    if buf.len() < 4 {
        return Err(GgufError::UnexpectedEof.into());
    }
    if &buf[0..4] != b"GGUF" {
        return Err(GgufError::BadMagic.into());
    }
    let mut pos: usize = 4;
    let version = read_u32(buf, &mut pos)?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version).into());
    }
    let tensor_count = read_u64(buf, &mut pos)?;
    let kv_count = read_u64(buf, &mut pos)?;
    if tensor_count > 100_000 || kv_count > 100_000 {
        return Err(GgufError::SizeOverflow.into());
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_string(buf, &mut pos)?;
        let vtype = read_u32(buf, &mut pos)?;
        let value = parse_value(buf, &mut pos, vtype)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(buf, &mut pos)?;
        let n_dims = read_u32(buf, &mut pos)?;
        if n_dims > 8 {
            return Err(GgufError::SizeOverflow.into());
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(buf, &mut pos)?);
        }
        let type_code = read_u32(buf, &mut pos)?;
        let ggml_type = GgmlType::from_u32(type_code);
        let offset = read_u64(buf, &mut pos)?;
        let mut n_elements: u64 = 1;
        for d in &dims {
            n_elements = n_elements.checked_mul(*d).ok_or(GgufError::SizeOverflow)?;
        }
        let (be, bb) = ggml_type.type_size().ok_or(GgufError::BadType(type_code))?;
        let byte_size = if be == 1 {
            n_elements.checked_mul(bb).ok_or(GgufError::SizeOverflow)?
        } else {
            let blocks = n_elements / be;
            if n_elements % be != 0 {
                return Err(GgufError::SizeOverflow.into());
            }
            blocks.checked_mul(bb).ok_or(GgufError::SizeOverflow)?
        };
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
            byte_size,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| match v {
            MetadataValue::U32(x) => Some(*x),
            _ => None,
        })
        .unwrap_or(32);
    let alignment = if alignment == 0 { 32 } else { alignment };
    let data_offset = align_up(pos as u64, alignment as u64);

    Ok((metadata, tensors, data_offset))
}

fn align_up(x: u64, align: u64) -> u64 {
    if align == 0 {
        return x;
    }
    let r = x % align;
    if r == 0 {
        x
    } else {
        x + (align - r)
    }
}

fn parse_value(buf: &[u8], pos: &mut usize, vtype: u32) -> anyhow::Result<MetadataValue> {
    match vtype {
        0 => Ok(MetadataValue::U8(read_u8(buf, pos)?)),
        1 => Ok(MetadataValue::I8(read_i8(buf, pos)?)),
        2 => Ok(MetadataValue::U16(read_u16(buf, pos)?)),
        3 => Ok(MetadataValue::I16(read_i16(buf, pos)?)),
        4 => Ok(MetadataValue::U32(read_u32(buf, pos)?)),
        5 => Ok(MetadataValue::I32(read_i32(buf, pos)?)),
        6 => {
            let bits = read_u32(buf, pos)?;
            Ok(MetadataValue::F32(f32::from_bits(bits)))
        }
        7 => {
            let b = read_u8(buf, pos)?;
            Ok(MetadataValue::Bool(b != 0))
        }
        8 => {
            let s = read_string(buf, pos)?;
            Ok(MetadataValue::Str(s))
        }
        9 => {
            let elem_type = read_u32(buf, pos)?;
            if elem_type == 9 {
                return Err(GgufError::NestedArray.into());
            }
            if !is_valid_type(elem_type) {
                return Err(GgufError::BadType(elem_type).into());
            }
            let count = read_u64(buf, pos)?;
            let min_size = min_elem_size(elem_type);
            let remaining = (buf.len() as u64).saturating_sub(*pos as u64);
            if count > remaining / min_size.max(1) + 1 {
                return Err(GgufError::UnexpectedEof.into());
            }
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(parse_value(buf, pos, elem_type)?);
            }
            Ok(MetadataValue::Array(ArrayValue { elem_type, values }))
        }
        10 => Ok(MetadataValue::U64(read_u64(buf, pos)?)),
        11 => Ok(MetadataValue::I64(read_i64(buf, pos)?)),
        12 => {
            let bits = read_u64(buf, pos)?;
            Ok(MetadataValue::F64(f64::from_bits(bits)))
        }
        x => Err(GgufError::BadType(x).into()),
    }
}

fn min_elem_size(t: u32) -> u64 {
    match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        8 => 8,
        10 | 11 | 12 => 8,
        _ => 1,
    }
}

fn is_valid_type(t: u32) -> bool {
    t <= 12
}

fn read_u8(buf: &[u8], pos: &mut usize) -> Result<u8, GgufError> {
    if *pos >= buf.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let v = buf[*pos];
    *pos += 1;
    Ok(v)
}

fn read_i8(buf: &[u8], pos: &mut usize) -> Result<i8, GgufError> {
    Ok(read_u8(buf, pos)? as i8)
}

fn read_u16(buf: &[u8], pos: &mut usize) -> Result<u16, GgufError> {
    if *pos + 2 > buf.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let mut b = [0u8; 2];
    b.copy_from_slice(&buf[*pos..*pos + 2]);
    *pos += 2;
    Ok(u16::from_le_bytes(b))
}

fn read_i16(buf: &[u8], pos: &mut usize) -> Result<i16, GgufError> {
    Ok(read_u16(buf, pos)? as i16)
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, GgufError> {
    if *pos + 4 > buf.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[*pos..*pos + 4]);
    *pos += 4;
    Ok(u32::from_le_bytes(b))
}

fn read_i32(buf: &[u8], pos: &mut usize) -> Result<i32, GgufError> {
    Ok(read_u32(buf, pos)? as i32)
}

fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, GgufError> {
    if *pos + 8 > buf.len() {
        return Err(GgufError::UnexpectedEof);
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[*pos..*pos + 8]);
    *pos += 8;
    Ok(u64::from_le_bytes(b))
}

fn read_i64(buf: &[u8], pos: &mut usize) -> Result<i64, GgufError> {
    Ok(read_u64(buf, pos)? as i64)
}

fn read_string(buf: &[u8], pos: &mut usize) -> Result<String, GgufError> {
    let len = read_u64(buf, pos)?;
    if len > (buf.len() - *pos) as u64 {
        return Err(GgufError::UnexpectedEof);
    }
    let len = len as usize;
    let bytes = &buf[*pos..*pos + len];
    *pos += len;
    String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::BadUtf8)
}
