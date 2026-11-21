// Root build script: only plugin versions live here, modules apply them.
plugins {
    id("com.android.application") version "9.5.2" apply false
    id("com.android.library") version "9.5.2" apply false
    // AGP 9 compiles Kotlin itself (built-in Kotlin), so only the Compose
    // compiler plugin is applied on top of it.
    id("org.jetbrains.kotlin.plugin.compose") version "2.4.21" apply false
}
