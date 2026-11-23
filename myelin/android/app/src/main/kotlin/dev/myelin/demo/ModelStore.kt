package dev.myelin.demo

import android.content.Context
import java.io.File

/**
 * Resolves the model directory the Rust core expects: `model.gguf`,
 * `tokenizer.json` and `chat_template.jinja` must sit in the same folder.
 *
 * `tokenizer.json` and `chat_template.jinja` are shipped in assets and copied
 * to `filesDir` on first use. The GGUF is too large for the APK, so it is
 * pushed separately:
 *
 *   adb push models/LFM3-1.7B-Q4_K_M.gguf \
 *       /data/data/dev.myelin.demo/files/model.gguf
 */
object ModelStore {

    private const val MODEL = "model.gguf"
    private val ASSETS = listOf("tokenizer.json", "chat_template.jinja")

    /** Copy the small assets and return the model path, or null if missing. */
    fun prepare(context: Context): String? {
        val dir = context.filesDir
        for (name in ASSETS) {
            val out = File(dir, name)
            if (out.exists() && out.length() > 0) continue
            runCatching {
                context.assets.open(name).use { input ->
                    out.outputStream().use { output -> input.copyTo(output) }
                }
            }
        }
        val model = File(dir, MODEL)
        return if (model.exists() && model.length() > 0) model.absolutePath else null
    }

    fun hint(context: Context): String =
        "Положите модель: adb push models/LFM3-1.7B-Q4_K_M.gguf " +
            "${context.filesDir.absolutePath}/$MODEL"
}
