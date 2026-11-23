package dev.myelin.demo

import android.content.Context

/** Process-wide application context, set once in [MyelinApp]. */
object AppContext {
    @Volatile
    private var context: Context? = null

    fun set(value: Context) {
        context = value.applicationContext
    }

    fun get(): Context = context ?: error("AppContext is not initialised")
}
