#!/usr/bin/env bash
REGEX_FEATURE='^feature/(.*)$'
REGEX_MASTER='^master$'
if [ "$#" -ne 2 ] && [ "$#" -ne 1 ] ; then
    echo Error in set_travis_reference - not enoght arguments
    echo "Usage: set_travis_reference <package_name> (user/channel) "
    echo sets the environment variable REFERENCE with package_name/version@user/channel
    echo Where version is derived from the TRAVIS_BRANCH/TRAVIS_TAG
elif [ ${#TRAVIS_TAG} -gt 0 ] ; then
    export CONAN_REFERENCE=$1/${TRAVIS_TAG}@$2
elif [[ $TRAVIS_BRANCH =~ $REGEX_MASTER ]] ; then
    export CONAN_REFERENCE=$1/latest@$2
elif [[ $TRAVIS_BRANCH =~ $REGEX_FEATURE ]]; then
    export CONAN_REFERENCE=$1/${BASH_REMATCH[1]}@$2;
else
    echo Error in set_travis_reference
    echo Expected either:
    echo 1:  a TRAVIS_TAG with a version number
    echo 2:  a TRAVIS_BRANCH with "feature/.*" or "master"
fi
