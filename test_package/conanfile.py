import os
import platform
from pathlib import Path
from conans import ConanFile, tools
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps

# This is a "hello world" type test that checks that the conan package can be consumed
# i.e. that that cmake support works, consumption of HDILib headers (compiler) and lib (linker) works
# and nothing else. It is is not a full unit or regression test for HDILib.


class HDILibTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    # cmake_paths locates the HDILib built with build_type None
    # cmake_find_package creates the lz4Targets.cmake locating - only Release needs to be configured
    generators = "CMakeDeps"

    def generate(self):
        if os.getenv("Analysis", None) is not None:
            return
        tc = CMakeToolchain(self)
        tc.variables["HDILib_ROOT"] = Path(
            self.deps_cpp_info["HDILib"].rootpath
        ).as_posix()
        tc.variables["flann_ROOT"] = Path(
            self.deps_cpp_info["flann"].rootpath
        ).as_posix()
        tc.variables["lz4_ROOT"] = Path(
            self.deps_cpp_info["lz4"].rootpath
        ).as_posix()
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def requirements(self):
        if os.getenv("Analysis", None) is not None:
            return
        print("In requirements")
        if self.settings.build_type == "None":
            print("Skip test_package requirements for build_type NONE")
            return
        else:
            print(
                f"Setting test_package requirements for build-type {self.settings.build_type}"
            )
            self.requires.add("lz4/1.9.2")

    def system_requirements(self):
        if os.getenv("Analysis", None) is not None:
            return
        if tools.os_info.is_linux:
            installer = tools.SystemPackageTool()
            installer.install("libomp5")
            installer.install("libomp-dev")

    def build(self):
        if os.getenv("Analysis", None) is not None:
            return
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    # def imports(self):
    #    self.copy("*.dll", dst="bin", src="bin")
    #    self.copy("*.dylib*", dst="bin", src="lib")
    #    self.copy("*.so*", dst="bin", src="lib")

    def test(self):
        if os.getenv("Analysis", None) is not None:
            return
        if not tools.cross_building(self.settings):
            os.chdir("bin")
            if platform.system() == "Windows":
                examplePath = Path("./", str(self.build_folder), "bin", "example.exe")
                self.run(f"{str(examplePath)}")
            else:
                self.run(".%sexample" % os.sep)
