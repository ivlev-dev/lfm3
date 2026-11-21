package dev.myelin.demo

import android.app.Application

/**
 * The Rust core is packaged as `libmyelin.so`, while the generated UniFFI
 * bindings default to `libuniffi_myelin.so`. The generated loader exposes an
 * override hook for exactly this case; point it at the real library name
 * before any UniFFI class is touched.
 */
class MyelinApp : Application() {

    override fun onCreate() {
        super.onCreate()
        System.setProperty(UNIFFI_LIBRARY_OVERRIDE, NATIVE_LIBRARY_NAME)
    }

    private companion object {
        const val UNIFFI_LIBRARY_OVERRIDE = "uniffi.component.myelin.libraryOverride"
        const val NATIVE_LIBRARY_NAME = "myelin"
    }
}
