# -*- coding: utf-8 -*-

from conans import ConanFile, CMake, tools
import os
import json


class HDILibConan(ConanFile):
    name = "HDILib"
    version = "1.0.0-alpha1"
    description = "HDILib is a library for the scalable analysis of large and high-dimensional data. "
    topics = ("embedding", "analysis", "n-dimensional", "tSNE")
    url = "https://github.com/biovault/HDILib"
    branch = "master"
    author = "B. van Lew <b.van_lew@lumc.nl>" #conanfile author
    license = "MIT"  # License for packaged library; please use SPDX Identifiers https://spdx.org/licenses/
    exports = ["LICENSE.md"]      # Packages the license for the conanfile.py
    generators = "cmake"

    # Options may need to change depending on the packaged library
    settings = {"os": None, "build_type": None, "compiler": None, "arch": None}
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": True, "fPIC": True}
    export_sources = "CMakeLists.txt", "hdi/*"

    _source_subfolder = name
    requires = (
        "CRoaring/0.2.63@lkeb/stable"
    )
    # Flann builds are bit complex and certain versions fail with 
    # certain platform, and compiler combinations. Hence use 
    # either self built 1.8.5 for Windows or system supplied 
    # 1.8.4 on Linux and Macos

    def system_requirements(self):
        if tools.os_info.is_linux:
            if tools.os_info.with_apt:
                installer = tools.SystemPackageTool()
                installer.install('libflann-dev')
        if tools.os_info.is_macos: 
            installer = tools.SystemPackageTool()    
            installer.install('libomp')
            installer.install('flann')
               
    def requirements(self):
        if self.settings.os == "Windows":
            self.requires("flann/1.8.5@lkeb/stable")
            
        
    def config_options(self):
        if self.settings.os == 'Windows':
            del self.options.fPIC 
    
    def source(self):
        source_url = self.url
        self.run("git clone {0}.git".format(self.url))
        os.chdir("./{0}".format(self._source_subfolder))
        self.run("git checkout {0}".format(self.branch))
        
        # Patch CMakeLists.txt to support /MT /MD linkage in MSVC if 
        # the packaged project doesn't have variables to set it properly
        conanproj = ("PROJECT(${PROJECT})\n"
                "include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)\n"
                "conan_basic_setup()\n"
        )
        os.chdir("..")
        tools.replace_in_file("HDILib/CMakeLists.txt", "PROJECT(${PROJECT})", conanproj)

    def _configure_cmake(self):
        cmake = CMake(self)
        if self.settings.os == "Windows" and self.options.shared:
            cmake.definitions["CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS"] = True
        if self.settings.os == "Linux" or self.settings.os == "Macos":
            cmake.definitions["CMAKE_CXX_STANDARD"] = 14
            cmake.definitions["CMAKE_CXX_STANDARD_REQUIRED"] = "ON"
        cmake.configure(source_folder=self._source_subfolder)
        cmake.verbose = True
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def package(self):
        self.copy(pattern="LICENSE", dst="licenses", src=self._source_subfolder)
        # If the CMakeLists.txt has a proper install method, the steps below may be redundant
        # If so, you can just remove the lines below
        self.copy(pattern="*", src=os.path.join(self.install_dir, 'Release'))


    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
