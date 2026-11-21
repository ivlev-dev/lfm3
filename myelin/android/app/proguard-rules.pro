# Release builds are not minified yet. If `isMinifyEnabled` is turned on, keep
# the generated UniFFI surface and JNA: both are reached through native callbacks
# and reflection, which R8 cannot see.
-keep class uniffi.** { *; }
-keep class com.sun.jna.** { *; }
-keepclassmembers class * implements uniffi.myelin.TokenSink { *; }
