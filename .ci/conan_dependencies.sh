if [ $# -eq 0 ] ; then
    project=.
else
    project=$1
fi
pip3 install --upgrade conan
conan user
cd $project && .ci/pemsetup.sh
conan remote add lkeb-artifactory $CONAN_UPLOAD
conan remote list
echo TRAVIS OS name $TRAVIS_OS_NAME

if [[ $TRAVIS_OS_NAME -eq osx ]] ; then
    CONAN_OS=Macos
    CONAN_COMPILER=apple-clang
fi

if [[ $TRAVIS_OS_NAME -eq linux ]] ; then
    export CONAN_OS=Linux
    export CONAN_COMPILER=gcc
fi
conan profile new action_build --force
conan profile update settings.os=$CONAN_OS action_build
conan profile update settings.os_build=$CONAN_OS action_build
conan profile update settings.arch=x86_64 action_build
conan profile update settings.arch_build=x86_64 action_build
conan profile update settings.compiler=$CONAN_COMPILER action_build
conan profile update settings.compiler.version=$CONAN_COMPILERVERSION action_build
conan profile update settings.compiler.libcxx=$CONAN_LIBC action_build
conan profile show action_build

conan install . -s build_type=Release --profile action_build
conan install . -s build_type=Debug --profile action_build
cd test_package
conan install . -s build_type=Release --profile action_build