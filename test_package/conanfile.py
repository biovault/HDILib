import os
import platform
from pathlib import Path
from conans import ConanFile, tools
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps
import shutil

required_conan_version = "~=1.64.0"

# This is a "hello world" type test that checks that the conan package can be consumed
# i.e. that that cmake support works, consumption of HDILib headers (compiler) and lib (linker) works
# and nothing else. It is is not a full unit or regression test for HDILib.


class HDILibTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    # cmake_paths locates the HDILib built with build_type None

    def generate(self):

        # deps = CMakeDeps(self)
        # deps.generate()

        if os.getenv("Analysis", None) is not None:
            return
        tc = CMakeToolchain(self)
        tc.variables["HDILib_ROOT"] = Path(
            self.deps_cpp_info["HDILib"].rootpath
        ).as_posix()
        # Use the cmake export in the flann package
        tc.variables["flann_ROOT"] = Path(
            self.deps_cpp_info["flann"].rootpath, "lib", "cmake"
        ).as_posix()
        # Use the cmake export in the lz4 package
        tc.variables["lz4_ROOT"] = Path(
            self.deps_cpp_info["lz4"].rootpath, "lib", "cmake"
        ).as_posix()
        tc.generate()

    def requirements(self):
        if os.getenv("Analysis", None) is not None:
            return
        print("In requirements")
        if self.settings.build_type == "None":
            print("Skip test_package requirements for build_type NONE")
            return
        else:
            self.requires("flann/1.9.2@lkeb/stable")

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
                shutil.copy(
                    Path(
                        self.deps_cpp_info["lz4"].rootpath, "bin", "Release", "lz4.dll"
                    ),
                    Path("./", str(self.build_folder), "bin"),
                )
                examplePath = Path("./", str(self.build_folder), "bin", "example.exe")
                self.run(f"{str(examplePath)}")
            elif platform.system() == "Darwin":
                shutil.copy(
                    Path(
                        self.deps_cpp_info["flann"].rootpath,
                        "lib",
                        "Release",
                        "libflann_cpp.1.9.dylib",
                    ),
                    Path("./"),
                )
            else:
                self.run(".%sexample" % os.sep)
