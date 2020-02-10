if (MSVC)
  set(HDI_PROGRAM_FILES_X86 "ProgramFiles(x86)") 
  find_program(HDI_INCREDIBUILD_EXE BuildConsole.exe PATHS
    "$ENV{ProgramFiles}/Xoreax/IncrediBuild"
    "$ENV{${SELX_PROGRAM_FILES_X86}}/Xoreax/IncrediBuild"
    DOC "Optional file path of IncrediBuild BuildConsole"
  )
endif()

function(SET_INCREDIBUILD_COMMAND proj)
  if (FOUND_INCREDIBUILD_EXE)
    set(INCREDIBUILD_COMMAND "${FOUND_INCREDIBUILD_EXE}" ${proj}.sln "/build" "$(Configuration)|$(Platform)" "/UseIDEMonitor" PARENT_SCOPE) 
  endif()
endfunction()
