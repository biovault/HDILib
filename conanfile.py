# -*- coding: utf-8 -*-

from conans import ConanFile, tools
from conan.tools.cmake import CMakeDeps, CMake, CMakeToolchain
from conans.tools import os_info, SystemPackageTool
import os
import sys
from packaging import version
from pathlib import Path

required_conan_version = "==1.62.0"

class HDILibConan(ConanFile):
    name = "HDILib"
    version = "1.3.0"
    description = "HDILib is a library for the scalable analysis of large and high-dimensional data. "
    topics = ("embedding", "analysis", "n-dimensional", "tSNE")
    url = "https://github.com/biovault/HDILib"
    author = "B. van Lew <b.van_lew@lumc.nl>"  # conanfile author
    license = "MIT"  # License for packaged library; please use SPDX Identifiers https://spdx.org/licenses/
    default_user = "lkeb"
    default_channel = "stable"

    # Options may need to change depending on the packaged library
    settings = "os", "compiler", "arch", "build_type"
    # Note : This should only be built with: shared=False, fPIC=True
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # scm = {
    #    "type": "git",dir
    #    "url": "https://github.com/biovault/HDILib.git",
    #    "submodule": "recursive"
    # }
    # , "conanbuildinfo.txt", "conanbuildinfo_debug.cmake", "conanbuildinfo_release.cmake", "conanbuildinfo_multi.cmake"
    exports = "hdi*", "external*", "cmake*", "CMakeLists.txt", "LICENSE"

    def _get_python_cmake(self):
        if None is not os.environ.get("APPVEYOR", None):
            pypath = Path(sys.executable)
            cmakePath = Path(pypath.parents[0], "Scripts/cmake.exe")
            return cmakePath
        return "cmake"

    # def system_requirements(self):
    #    if tools.os_info.is_macos:
    #        target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "10.13")
    #        if version.parse(target) > version.parse("10.12"):
    #            installer = tools.SystemPackageTool()
    #            installer.install("libomp")

    def requirements(self):
        if self.settings.build_type == "None":
            print("Skip root package requirements for build_type NONE")
            return
        #self.requires("flann/1.9.2")
        self.requires("flann/1.9.2@lkeb/stable")

    def system_requirements(self):
        if os_info.is_macos:
            installer = SystemPackageTool()
            installer.install('libomp')

    def configure(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def generate(self):
        print("In generate")
        generator = None
        if self.settings.os == "Macos":
            generator = "Xcode"

        if self.settings.os == "Linux":
            generator = "Ninja Multi-Config"

        cmake = CMakeDeps(self)
        cmake.generate()

        # A toolchain file can be used to modify CMake variables
        tc = CMakeToolchain(self)
        if self.settings.os == "Windows":
            tc.variables["CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS"] = True
        tc.variables["HDILib_VERSION"] = self.version
        if self.build_folder is not None:
            tc.variables["CMAKE_INSTALL_PREFIX"] = str(
                Path(self.build_folder, "install").as_posix()
            )
        else:
            tc.variables["CMAKE_INSTALL_PREFIX"] = "${CMAKE_BINARY_DIR}"
        tc.variables["ENABLE_CODE_ANALYSIS"] = "ON"
        tc.variables["CMAKE_VERBOSE_MAKEFILE"] = "ON"
        if os.getenv("Analysis", None) is None:
            tc.variables["ENABLE_CODE_ANALYSIS"] = "OFF"
        tc.variables[
            "CMAKE_MSVC_RUNTIME_LIBRARY"
        ] = "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
        print("Call toolchain generate")
        tc.generate()

    def _configure_cmake(self):
        cmake = CMake(self)
        print(f"Set version to {self.version}")
        cmake.configure()
        return cmake

    def build(self):
        print(
            f"Build folder {self.build_folder} \n Package folder {self.package_folder}\n Source folder {self.source_folder}"
        )
        install_dir = Path(self.build_folder).joinpath("install")
        install_dir.mkdir(exist_ok=True)

        cmake_debug = self._configure_cmake()
        cmake_debug.build(build_type="Debug")
        cmake_debug.install(build_type="Debug")

        if os.getenv("Analysis", None) is None:
            # Disable code analysis in Release mode
            cmake_release = self._configure_cmake()
            cmake_release.build(build_type="Release")
            cmake_release.install(build_type="Release")

    def package_id(self):
        # The package contains both Debug and Release build types
        del self.info.settings.build_type
        if self.settings.compiler == "Visual Studio":
            del self.info.settings.compiler.runtime

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
        self.cpp_info.set_property("skip_deps_file", True)
        self.cpp_info.set_property("cmake_config_file", True)

    def package(self):
        install_dir = Path(self.build_folder).joinpath("install")
        self.copy(pattern="*", src=str(install_dir))
        # Add the debug support files to the package
        # (*.pdb) if building the Visual Studio version
        if self.settings.compiler == "Visual Studio":
            self.copy("*.pdb", dst="lib/Debug", keep_path=False)
