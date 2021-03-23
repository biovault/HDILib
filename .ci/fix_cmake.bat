echo ON
if exist "%REPLACE_CMAKE_PATH%" (
    echo Replacing cmake
    dir "%PYTHON_HOME%\Lib\site-packages\cmake\data\"
    xcopy "%PYTHON_HOME%\Lib\site-packages\cmake\data\." "%REPLACE_CMAKE_PATH%" /s /e /y /q
    dir "%REPLACE_CMAKE_PATH%"
    "%REPLACE_CMAKE_PATH%bin\cmake" --version
) else (
    echo Using system CMake
)