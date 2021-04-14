Param(
    [Parameter(Mandatory=$true)]
    [String] $PackageName,

    [Parameter(Mandatory=$false)]
    [String] $UserChannel
)

$REGEX_FEATURE='^feature/(.*)$'
$REGEX_MASTER='^master$'
$branch=[Environment]::GetEnvironmentVariable('APPVEYOR_REPO_BRANCH')

if ( [Environment]::GetEnvironmentVariable('APPVEYOR_REPO_TAG')) {
    $reference=$PackageName+'/'+[Environment]::GetEnvironmentVariable('APPVEYOR_REPO_TAG_NAME')+'@'+$UserChannel
    [System.Environment]::SetEnvironmentVariable('CONAN_REFERENCE',$reference)
}
elseif ($branch -match $REGEX_MASTER) {
    $reference=$PackageName+'/latest@'+$UserChannel
    [System.Environment]::SetEnvironmentVariable('CONAN_REFERENCE',$reference)
}
elseif ($branch -match $REGEX_FEATURE) {
    $reference=$PackageName+'/'+$Matches.1+'@'+$UserChannel
    [System.Environment]::SetEnvironmentVariable('CONAN_REFERENCE',$reference)
}
else {
    echo "Error in set_appveyor_conan_reference"
    echo "Expected either:"
    echo "1:  a APPVEYOR_REPO_TAG_NAME with a version number"
    echo "2:  a APPVEYOR_REPO_BRANCH with 'feature/.*' or 'master'"
}