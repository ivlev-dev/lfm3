package dev.myelin.demo

import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import uniffi.myelin.ChatMessage
import uniffi.myelin.Engine
import uniffi.myelin.MyelinException
import uniffi.myelin.Sampling
import uniffi.myelin.Session
import uniffi.myelin.TokenEvent
import uniffi.myelin.TokenSink

/**
 * Lifecycle-aware wrapper around the generated UniFFI objects.
 *
 * `Session.generate` is a *blocking* call that pushes [TokenEvent]s into a
 * [TokenSink] callback, so it is bridged into a cold [Flow] with `callbackFlow`
 * and executed on [Dispatchers.IO]. Cancelling the collector (or calling
 * [cancel]) makes the Rust side stop at the next token boundary.
 *
 * `Engine(modelPath = null)` selects the in-process stub backend; a path to a
 * GGUF whose folder also holds `tokenizer.json` and `chat_template.jinja`
 * loads the real CPU backend.
 */
class MyelinClient(modelPath: String? = null) : AutoCloseable {

    private val engine: Engine = Engine(modelPath = modelPath)
    private val session: Session = engine.newSession()
    private val closed = AtomicBoolean(false)

    fun stream(
        messages: List<ChatMessage>,
        sampling: Sampling = Sampling(),
    ): Flow<TokenEvent> = callbackFlow {
        val sink = object : TokenSink {
            override fun onToken(event: TokenEvent) {
                // Backpressure instead of dropping tokens; the caller is a
                // dedicated IO worker, so blocking here is safe and cheap.
                trySendBlocking(event)
            }
        }

        val worker = launch(Dispatchers.IO) {
            try {
                session.generate(messages, sampling, sink)
                close()
            } catch (cancelled: MyelinException.Cancelled) {
                // Cooperative stop: close the flow normally, not exceptionally.
                close()
            } catch (error: Throwable) {
                close(error)
            }
        }

        awaitClose {
            session.cancel()
            worker.cancel()
        }
    }

    /** Asks the running generation to stop; [stream] then completes cleanly. */
    fun cancel() {
        if (!closed.get()) {
            session.cancel()
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        session.close()
        engine.close()
    }
}
