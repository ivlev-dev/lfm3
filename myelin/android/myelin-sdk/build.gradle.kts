import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.library")
}

android {
    namespace = "dev.myelin.sdk"
    compileSdk = 37
    compileSdkMinor = 1
    buildToolsVersion = "37.0.0"

    defaultConfig {
        minSdk = 26
        // The Rust core only ships an arm64 build today.
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources {
        // Native code is never compressed so it can be mmap'ed straight from the APK.
        noCompress.add("so")
    }

    packaging {
        jniLibs {
            // AGP keeps 16 KB-aligned, uncompressed .so files (required by Android 15+).
            useLegacyPackaging = false
        }
    }
}

kotlin {
    jvmToolchain(17)
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // UniFFI Kotlin bindings call into Rust through JNA. Android needs the AAR
    // (it carries libjnidispatch.so); the plain JAR does not.
    // `api`, because generated public signatures mention `com.sun.jna.Pointer`.
    api("net.java.dev.jna:jna:5.19.1@aar")
}
