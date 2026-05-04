# HDILib example
Minimal setup for using the HDILib for computing a t-SNE embedding.

When building, tell CMake where to find the library with `HDILIB_ROOT` (`"PATH_TO/HDILib_install/lib/cmake/HDILib"`).

When using vcpkg to build dependencies, be sure to define `VCPKG_MANIFEST_DIR` as `"PATH_TO/HDILib"` and when using Windows define `VCPKG_TARGET_TRIPLET` as `x64-windows-static-md`

