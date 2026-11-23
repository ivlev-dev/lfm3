package dev.myelin.demo

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import uniffi.myelin.ChatMessage
import uniffi.myelin.MyelinException
import uniffi.myelin.Sampling
import uniffi.myelin.TokenEvent

data class UiMessage(
    val role: Role,
    val text: String,
) {
    enum class Role { User, Assistant }
}

data class ChatUiState(
    val prompt: String = "",
    val messages: List<UiMessage> = emptyList(),
    val isGenerating: Boolean = false,
    val error: String? = null,
)

class ChatViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private val clientResult: Result<MyelinClient> = runCatching {
        val context = AppContext.get()
        val path = ModelStore.prepare(context) ?: error(ModelStore.hint(context))
        MyelinClient(path)
    }
    private val client: MyelinClient? = clientResult.getOrNull()

    private var generationJob: Job? = null

    init {
        clientResult.exceptionOrNull()?.let { error ->
            _uiState.update { it.copy(error = error.toUserMessage()) }
        }
    }

    fun onPromptChange(value: String) {
        _uiState.update { it.copy(prompt = value) }
    }

    fun generate() {
        val myelin = client ?: return
        val snapshot = _uiState.value
        val prompt = snapshot.prompt.trim()
        if (prompt.isEmpty() || snapshot.isGenerating) return

        val history = snapshot.messages
            .filter { it.text.isNotBlank() }
            .map { it.asChatMessage() } +
            ChatMessage(role = ROLE_USER, content = prompt)

        _uiState.update { state ->
            state.copy(
                prompt = "",
                error = null,
                isGenerating = true,
                messages = state.messages +
                    UiMessage(UiMessage.Role.User, prompt) +
                    UiMessage(UiMessage.Role.Assistant, ""),
            )
        }

        generationJob = viewModelScope.launch {
            try {
                myelin.stream(messages = history, sampling = Sampling()).collect { event ->
                    when (event) {
                        is TokenEvent.Delta -> appendToAnswer(event.text)
                        is TokenEvent.ToolCall -> appendToAnswer("\n[tool] ${event.payload}\n")
                        is TokenEvent.Done -> Unit // stream ends right after Done
                        is TokenEvent.Error -> _uiState.update { it.copy(error = event.message) }
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (cancelled: MyelinException.Cancelled) {
                // The user pressed Stop: expected outcome, not an error.
            } catch (error: Throwable) {
                _uiState.update { it.copy(error = error.toUserMessage()) }
            } finally {
                _uiState.update { it.copy(isGenerating = false) }
                generationJob = null
            }
        }
    }

    /** Cooperative stop: generated tokens already shown are kept. */
    fun stop() {
        client?.cancel()
    }

    override fun onCleared() {
        generationJob?.cancel()
        client?.close()
        super.onCleared()
    }

    private fun appendToAnswer(chunk: String) {
        _uiState.update { state ->
            val last = state.messages.lastOrNull()
            if (last == null || last.role != UiMessage.Role.Assistant) {
                state
            } else {
                state.copy(
                    messages = state.messages.dropLast(1) + last.copy(text = last.text + chunk),
                )
            }
        }
    }

    private fun UiMessage.asChatMessage(): ChatMessage = ChatMessage(
        role = if (role == UiMessage.Role.User) ROLE_USER else ROLE_ASSISTANT,
        content = text,
    )

    private companion object {
        const val ROLE_USER = "user"
        const val ROLE_ASSISTANT = "assistant"
    }
}

/**
 * Generated bindings expose `MyelinException` (the Rust `MyelinError` enum);
 * turn it into something readable for the UI.
 */
private fun Throwable.toUserMessage(): String = when (this) {
    is MyelinException.Cancelled -> "Generation cancelled."
    is MyelinException.InvalidArgument -> "Invalid argument: ${message ?: "unknown"}"
    is MyelinException.InvalidState -> "Invalid state: ${message ?: "unknown"}"
    is MyelinException.Io -> "I/O error: ${message ?: "unknown"}"
    is MyelinException.Backend -> "Backend error: ${message ?: "unknown"}"
    is MyelinException.Internal -> "Internal error: ${message ?: "unknown"}"
    else -> message ?: javaClass.simpleName
}
