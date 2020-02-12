import os
import platform

from conans import ConanFile, CMake, tools

# This is a "hello world" type test that checks that the conan package
# links and little more. It is is not a full unit or regression
# test for HDILib.

class HDILibTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    # lz4/1.9.2 is in conan center
    requires = (
        "lz4/1.9.2"
    )
    
    def system_requirements(self):
        if tools.os_info.is_linux:
            installer = tools.SystemPackageTool()
            installer.install("libomp-dev")
            
    def build(self):
        cmake = CMake(self)
        # Current dir is "test_package/build/<build_id>" and CMakeLists.txt is
        # in "test_package"
        cmake.configure()
        cmake.build()

    def imports(self):
        self.copy("*.dll", dst="bin", src="bin")
        self.copy("*.dylib*", dst="bin", src="lib")
        self.copy('*.so*', dst='bin', src='lib')
        self.copy("HDILib", dst="HDILib", folder=True)

    def test(self):
        if not tools.cross_building(self.settings):
            os.chdir("bin")
            if platform.system() == 'Windows':
                self.run(".%sexample.exe" % os.sep)
            else:
                self.run(".%sexample" % os.sep)                
