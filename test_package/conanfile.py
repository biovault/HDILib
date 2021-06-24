import os
import platform
from pathlib import Path
import pathlib

from conans import ConanFile, CMake, tools

# This is a "hello world" type test that checks that the conan package can be consumed
# i.e. that that cmake support works, consumption of HDILib headers (compiler) and lib (linker) works
# and nothing else. It is is not a full unit or regression test for HDILib.

class HDILibTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    # cmake_paths locates the HDILib built with build_type None
    # cmake_find_package creates the lz4Targets.cmake locating - only Release needs to be configured
    generators = "cmake_find_package_multi", "cmake_paths"

    def configure(self):
        print("In configure")

    def config_options(self):
        print("In config_options")

    def requirements(self):
        print("In requirements")
        if self.settings.build_type == "None":
            print("Skip test_package requirements for build_type NONE")
            return
        else:
            print(f"Setting test_package requirements for build-type {self.settings.build_type }")
            self.requires.add("lz4/1.9.2")

    def system_requirements(self):
        if tools.os_info.is_linux:
            installer = tools.SystemPackageTool()
            installer.install("libomp-dev")

    def build(self):
        cmake = CMake(self)
        if self.settings.build_type == "None":
            print("Test consumption of HDILib in Release mode")
            self.settings.build_type = "Release"
            cmake = CMake(self, build_type="Release")
        print(f"")
        cmake.definitions["CMAKE_TOOLCHAIN_FILE"] = "conan_paths.cmake"
        cmake.configure()
        cmake.build()

    def imports(self):
        self.copy("*.dll", dst="bin", src="bin")
        self.copy("*.dylib*", dst="bin", src="lib")
        self.copy('*.so*', dst='bin', src='lib')

    def test(self):
        if not tools.cross_building(self.settings):
            os.chdir("bin")
            if platform.system() == 'Windows':
                examplePath = Path("./", str(self.build_folder), "bin", "example.exe")
                self.run(f"{str(examplePath)}")
            else:
                self.run(".%sexample" % os.sep)
