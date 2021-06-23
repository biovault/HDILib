Param(
    [Parameter(Mandatory=$true)]
    [String] $PackageName,

    [Parameter(Mandatory=$false)]
    [String] $UserChannel
)

$REGEX_FEATURE='^feature/(.*)$'
$REGEX_RELEASE='^feature/(.*)$'
$REGEX_MASTER='^master$'
$branch=[Environment]::GetEnvironmentVariable('APPVEYOR_REPO_BRANCH')

if ( [Environment]::GetEnvironmentVariable('APPVEYOR_REPO_TAG' -eq 'true')) {
    $reference=$PackageName+'/'+[Environment]::GetEnvironmentVariable('APPVEYOR_REPO_TAG_NAME')+'@'+$UserChannel
    [System.Environment]::SetEnvironmentVariable('CONAN_REFERENCE',$reference, 'User')
}
elseif ($branch -match $REGEX_MASTER) {
    $reference=$PackageName+'/latest@'+$UserChannel
    [System.Environment]::SetEnvironmentVariable('CONAN_REFERENCE',$reference, 'User')
}
elseif ($branch -match $REGEX_FEATURE) {
    $reference=$PackageName+'/'+$Matches.1+'@'+$UserChannel
    [System.Environment]::SetEnvironmentVariable('CONAN_REFERENCE',$reference, 'User')
}
elseif ($branch -match $REGEX_RELEASE) {
    $reference=$PackageName+'/'+$Matches.1+'@'+$UserChannel
    [System.Environment]::SetEnvironmentVariable('CONAN_REFERENCE',$reference, 'User')}
else {
    Write-Output "Error in set_appveyor_conan_reference"
    Write-Output "Expected either:"
    Write-Output "1:  a APPVEYOR_REPO_TAG_NAME with a version number"
    Write-Output "2:  a APPVEYOR_REPO_BRANCH with 'feature/.*' or 'master'"
}