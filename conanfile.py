# -*- coding: utf-8 -*-

from conans import ConanFile, CMake, tools
import os
import re
import subprocess
import sys
from packaging import version
from pathlib import Path

class HDILibConan(ConanFile):
    name = "HDILib"
    default_version = "1.0.0-alpha1"
    description = "HDILib is a library for the scalable analysis of large and high-dimensional data. "
    topics = ("embedding", "analysis", "n-dimensional", "tSNE")
    url = "https://github.com/biovault/HDILib"
    author = "B. van Lew <b.van_lew@lumc.nl>" #conanfile author
    license = "MIT"  # License for packaged library; please use SPDX Identifiers https://spdx.org/licenses/
    default_user = "lkeb"
    default_channel = "stable"

    generators = ["cmake_multi", "cmake_find_package_multi"]

    # Options may need to change depending on the packaged library
    settings = "os", "compiler", "arch", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": True, "fPIC": True}

    #scm = {
    #    "type": "git",dir
    #    "url": "https://github.com/biovault/HDILib.git",
    #    "submodule": "recursive"
    #}
    #, "conanbuildinfo.txt", "conanbuildinfo_debug.cmake", "conanbuildinfo_release.cmake", "conanbuildinfo_multi.cmake"
    exports = "hdi*", "external*", "cmake*", "*.cmake", "CMakeLists.txt", "LICENSE"

    # Flann builds are bit complex and certain versions fail with 
    # certain platform, and compiler combinations. Hence use 
    # either self built 1.8.5 for Windows or system supplied 
    # 1.8.4 on Linux and Macos

    def _get_python_cmake(self):
        if None is not os.environ.get("APPVEYOR", None):
            pypath = Path(sys.executable)
            cmakePath = Path(pypath.parents[0], "Scripts/cmake.exe")
            return cmakePath
        return "cmake"

    def system_requirements(self):
        if tools.os_info.is_macos:
            target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', '10.13')
            if version.parse(target) > version.parse('10.12'):
                installer = tools.SystemPackageTool()
                installer.install('libomp')

    def requirements(self):
        if self.settings.build_type == "None":
             print("Skip root package requirements for build_type NONE")
             return
        if self.settings.os == "Windows":
            self.requires("flann/1.8.5@lkeb/stable")
        else:
            # Macos and flann use 1.8.4
            self.requires("flann/1.8.4@lkeb/stable")
        # self.requires.add("lz4/1.9.2")

    def configure(self):
        if self.settings.compiler == "Visual Studio":
            del self.settings.compiler.runtime

    def config_options(self):
        if self.settings.os == 'Windows':
            del self.options.fPIC

    def _configure_cmake(self, build_type):
        # Inject the conan dependency paths into the CMakeLists.txt
        conanproj = ("PROJECT(${PROJECT})\n"
                "include(${CMAKE_BINARY_DIR}/conanbuildinfo_multi.cmake)\n"
                "conan_basic_setup()\n"
        )
        tools.replace_in_file("CMakeLists.txt", "PROJECT(${PROJECT})", conanproj)
        if self.settings.os == "Macos":
            cmake = CMake(self, generator='Xcode', build_type=build_type)
        else:
            cmake = CMake(self, build_type=build_type)
        if self.settings.os == "Windows" and self.options.shared:
            cmake.definitions["CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS"] = True
        if self.settings.os == "Linux" or self.settings.os == "Macos":
            cmake.definitions["CMAKE_CXX_STANDARD"] = 14
            cmake.definitions["CMAKE_CXX_STANDARD_REQUIRED"] = "ON"
        cmake.definitions["HDI_EXTERNAL_FLANN_INCLUDE_DIR"] = "${CONAN_INCLUDE_DIRS_FLANN}"
        cmake.definitions["HDI_USE_ROARING"] = "OFF"
        cmake.definitions["HDILib_VERSION"] = self.version
        print(f"Set version to {self.version}")
        cmake.configure()
        cmake.verbose = True
        return cmake

    def build(self):
        print(f"Build folder {self.build_folder} \n Package folder {self.package_folder}\n Source folder {self.source_folder}")
        install_dir = Path(self.build_folder).joinpath("install")
        install_dir.mkdir(exist_ok=True)
        config = str(self.settings.build_type)
        print(f"Installing: install for {config} build")
        cmakepath = self._get_python_cmake()

        cmake_debug = self._configure_cmake('Debug')
        cmake_debug.build()
        result = subprocess.run([f"{str(cmakepath)}",
                        "--install", self.build_folder,
                        "--config", "Debug",
                        "--verbose",
                        "--prefix", str(install_dir)], capture_output=True)

        cmake_release = self._configure_cmake('Release')
        cmake_release.build()
        result = subprocess.run([f"{str(cmakepath)}",
                        "--install", self.build_folder,
                        "--config", "Release",
                        "--verbose",
                        "--prefix", str(install_dir)], capture_output=True)


        print(f"Install for {config} build - complete. \n Output: {result.stdout} \n Errors: {result.stderr}")

    def package(self):
        install_dir = Path(self.build_folder).joinpath("install")
        self.copy(pattern="*", src=str(install_dir))
        # Add the debug support files to the package
        # (*.pdb) if building the Visual Studio version
        if self.settings.compiler == "Visual Studio":
            self.copy("*.pdb", dst="lib/Debug", keep_path=False)

    def package_id(self):
        # The package contains both Debug and Release build types
        del self.info.settings.build_type
        if self.settings.compiler == "Visual Studio":
            del self.settings.compiler.runtime

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
