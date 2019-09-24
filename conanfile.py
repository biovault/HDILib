# -*- coding: utf-8 -*-

from conans import ConanFile, CMake, tools
import os
import json


class HDILibConan(ConanFile):
    name = "HDILib"
    version = "1.0.0-alpha1"
    description = "HDILib is a library for the scalable analysis of large and high-dimensional data. "
    # topics can get used for searches, GitHub topics, Bintray tags etc. Add here keywords about the library
    topics = ("conan", "analysis", "n-dimensional", "tSNE")
    url = "https://github.com/biovault/HDILib"
    branch = "master"
    author = "B. van Lew <b.van_lew@lumc.nl>"
    license = "MIT"  # Indicates license type of the packaged library; please use SPDX Identifiers https://spdx.org/licenses/
    exports = ["LICENSE.md"]      # Packages the license for the conanfile.py
    # Remove following lines if the target lib does not use cmake.
    generators = "cmake"

    # Options may need to change depending on the packaged library
    settings = {"os": None, "build_type": None, "compiler": None, "arch": None}
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": True, "fPIC": True}
    export_sources = "CMakeLists.txt", "hdi/*"

    _source_subfolder = name
    # requires = (
        # "qt/5.12.4@lkeb/stable",
        # "bzip2/1.0.8@conan/stable"
    # )

    def system_requirements(self):
        # if tools.os_info.is_linux:
            # if tools.os_info.with_apt:
                # installer = tools.SystemPackageTool()
                # installer.install('mesa-common-dev')
                # installer.install('libgl1-mesa-dev')
                # installer.install('libxcomposite-dev')
                # installer.install('libxcursor-dev')
                # installer.install('libxi-dev')
                # installer.install('libnss3-dev')
                # installer.install('libnspr4-dev')
                # installer.install('libfreetype6-dev')
                # installer.install('libfontconfig1-dev')
                # installer.install('libxtst-dev')
                # installer.install('libasound2-dev')
                # installer.install('libdbus-1-dev')
        if tools.os_info.is_macos: 
            installer = tools.SystemPackageTool()    
            installer.install('libomp')              
                
    def config_options(self):
        if self.settings.os == 'Windows':
            del self.options.fPIC 
    
    def source(self):
        source_url = self.url
        self.run("git clone {0}.git".format(self.url))
        os.chdir("./{0}".format(self._source_subfolder))
        self.run("git checkout {0}".format(self.branch))
        
        # This small hack might be useful to guarantee proper /MT /MD linkage
        # in MSVC if the packaged project doesn't have variables to set it
        # properly
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
        cmake.configure(source_folder=".")
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
