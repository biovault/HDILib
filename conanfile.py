# -*- coding: utf-8 -*-

from conans import ConanFile, CMake, tools
import os
import json
import re

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
        "url": "auto",
        "revision": "auto",
        "submodule": "recursive"
    }
    exports = "hdi*", "CMakeLists.txt", "LICENSE", 
    requires = (
        "CRoaring/0.2.63@lkeb/stable",
    )
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
        git = tools.Git(folder=self.recipe_folder)
        branch = git.get_branch()
        
        rel_match = re.compile("release/(\d+\.\d+.\d+)(.*)")
        feat_match = re.compile("feature/(.*)")
        self.version = self.default_version
        if branch == 'master':
            self.version = 'latest'
        else: 
            rel = rel_match(branch)
            if rel is not None:
                self.version = rel.group(1) + rel.group(2)
            else: 
                feat = feat_match(branch)
                if feat is not None:
                    self.version = 'latest_feat_' + feat.group(1)

        
    def system_requirements(self):
        if tools.os_info.is_macos: 
            
            installer = tools.SystemPackageTool()  
            installer.install('llvm')
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
        cmake.configure()
        cmake.verbose = True
        return cmake

    def build(self):
        print('Build folder: ', self.build_folder)
        print(os.listdir(self.build_folder))
        cmake = self._configure_cmake()
        cmake.build()

    def package(self):
        self.copy(pattern="LICENSE", dst="licenses", src=self.source_folder)
        # If the CMakeLists.txt has a proper install method, the steps below may be redundant
        # If so, you can just remove the lines below
        self.copy("*.h", dst="include", keep_path=True)
        self.copy("*.hpp", dst="include", keep_path=True)       
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.dylib", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy(pattern="*.pdb", dst="bin", keep_path=False)


    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
