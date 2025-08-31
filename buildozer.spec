[app]
title = Twist Angle Detector
package.name = twistangledetector
package.domain = org.example
source.dir = .
source.include_exts = py,kv,png,jpg,atlas
version = 0.1
requirements = python3,kivy,opencv,numpy
orientation = landscape
fullscreen = 0
android.permissions = CAMERA,INTERNET
android.api = 31
android.minapi = 21
android.archs = arm64-v8a,armeabi-v7a
# speed up buildozer/python-for-android behavior
p4a.branch = master
# optional: reduce size
# android.ndk_api = 21

[buildozer]
log_level = 2

[app:android]
# Leave blank unless you know you need custom recipes/deps
