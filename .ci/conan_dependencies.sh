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
conan profile new action_build --force --detect && conan profile show action_build
conan install . -s build_type=Release --profile action_build
conan install . -s build_type=Debug --profile action_build
cd test_package
conan install . -s build_type=Release --profile action_build