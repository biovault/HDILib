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

    generators = "cmake"

    # Options may need to change depending on the packaged library
    settings = "os", "build_type", "compiler", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": True, "fPIC": True}

    scm = {
        "type": "git",
        "url": "https://github.com/biovault/HDILib.git",
        "submodule": "recursive"
    }
    exports = "hdi*", "CMakeLists.txt", "LICENSE", 

    # Flann builds are bit complex and certain versions fail with 
    # certain platform, and compiler combinations. Hence use 
    # either self built 1.8.5 for Windows or system supplied 
    # 1.8.4 on Linux and Macos

    # Set version based on branch according to the following:
    #
    # master - gets version "latest"
    # release/x.y.z - gets version "x.y.z"
    # feature/blahblah - gets version "latest_feat_blahblah
    # otherwise use the self.version hardcoded is used
    def set_version(self):
        ci_branch = os.getenv("CONAN_HDILIB_CI_BRANCH", "master")

        print("Building branch: ", ci_branch) 
        rel_match = re.compile("release/(\d+\.\d+.\d+)(.*)")
        feat_match = re.compile("feature/(.*)")

        if ci_branch == "master":
            self.version = "latest"
        else:
            rel = rel_match.search(ci_branch)
            if rel is not None:
                self.version = rel.group(1) + rel.group(2)
            else:
                feat = feat_match.search(ci_branch)
                if feat is not None:
                    self.version = feat.group(1)
        self.scm["revision"] = ci_branch

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
        if self.settings.os == "Windows":
            self.requires("flann/1.8.5@lkeb/stable")
        else:
            # Macos and flann use 1.8.4
            self.requires("flann/1.8.4@lkeb/stable")

    def config_options(self):
        if self.settings.os == 'Windows':
            del self.options.fPIC

    def _configure_cmake(self):
        # Inject the conan dependency paths into the CMakeLists.txt
        conanproj = ("PROJECT(${PROJECT})\n"
                "include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)\n"
                "conan_basic_setup()\n"
        )
        tools.replace_in_file("CMakeLists.txt", "PROJECT(${PROJECT})", conanproj)
        if self.settings.os == "Macos":
            cmake = CMake(self, generator='Xcode')
        else:
            cmake = CMake(self)
        if self.settings.os == "Windows" and self.options.shared:
            cmake.definitions["CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS"] = True
        if self.settings.os == "Linux" or self.settings.os == "Macos":
            cmake.definitions["CMAKE_CXX_STANDARD"] = 14
            cmake.definitions["CMAKE_CXX_STANDARD_REQUIRED"] = "ON"
        cmake.definitions["HDI_EXTERNAL_FLANN_INCLUDE_DIR"] =  "${CONAN_INCLUDE_DIRS_FLANN}"
        cmake.definitions["HDI_USE_ROARING"] = "OFF"
        cmake.definitions["HDILib_VERSION"] = self.version
        print(f"Set version to {self.version}")
        cmake.configure()
        cmake.verbose = True
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        install_dir = Path(self.build_folder).joinpath("install")
        install_dir.mkdir(exist_ok=True)
        config = str(self.settings.build_type)
        print(f"Installing: install for {config} build")
        cmakepath = self._get_python_cmake()
        result = subprocess.run([f"{str(cmakepath)}",
                        "--install", self.build_folder,
                        "--config", config,
                        "--verbose",
                        "--prefix", str(install_dir)], capture_output=True)
        print(f"Install for {config} build - complete. \n Output: {result.stdout} \n Errors: {result.stderr}")

    def package(self):
        install_dir = Path(self.build_folder).joinpath("install")
        self.copy(pattern="*", src=str(install_dir))

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
