# Writes a header file that includes all of the basic functionality
# for the expression library.  This can be used rather than including
# various other header files

include_directories( ${CMAKE_BINARY_DIR}/tmp )

# --------------------------------------------------------

set( FMHeader ${CMAKE_BINARY_DIR}/tmp/expression/FieldManager.h )
file( WRITE ${FMHeader} "#ifndef Expr_FieldManager_h\n#define Expr_FieldManager_h\n" )
if( ENABLE_UINTAH )
  file( APPEND ${FMHeader} "\n#define ENABLE_UINTAH\n\n" )
  file( APPEND ${FMHeader} "#include <expression/uintah/UintahFieldManager.h>\n" )
elseif( ENABLE_HIMALAYA )
  file( APPEND ${FMHeader} "#include <expression/himalaya/HimalayaFieldManager.h>\n" )
elseif( ENABLE_LBMS )
  file( APPEND ${FMHeader} "#include <expression/lbms/LBMSFieldManager.h>\n" )
else()
  file( APPEND ${FMHeader} "#include <expression/DefaultFieldManager.h>\n" )
endif()
set( FMGRStruct "
  template< typename FieldT >
  struct FieldMgrSelector{
" )
if( ENABLE_UINTAH )
  set( FMGRStruct   "${FMGRStruct}    typedef UintahFieldManager<FieldT> type;\n" )
elseif( ENABLE_HIMALAYA )
  set( FMGRStruct   "${FMGRStruct}    typedef HimalayaFieldManager<FieldT> type;\n" )
elseif( ENABLE_LBMS )
  set( FMGRStruct   "${FMGRStruct}    typedef LBMSFieldManager<FieldT> type;\n" )
else()
  set( FMGRStruct   "${FMGRStruct}    typedef DefaultFieldManager<FieldT> type;\n" )
endif()
set( FMGRStruct "${FMGRStruct}  };\n" )
file( APPEND ${FMHeader} "\nnamespace Expr{
${FMGRStruct}
}  // namespace Expr\n
#endif // Expr_FieldManager_h\n" )
  
# --------------------------------------------------------

message(STATUS "Writing ExprLib.h header file")

set( ExprHeader ${CMAKE_BINARY_DIR}/tmp/expression/ExprLib.h )
file( WRITE ${ExprHeader}
  "#ifndef ExprLib_h
#define ExprLib_h\n\n"
  )

file( APPEND ${ExprHeader}
  "#include <expression/FieldManager.h>       // field manager definition \n"
  )
if( ENABLE_UINTAH )
  file( APPEND ${ExprHeader} "\n#define ENABLE_UINTAH\n\n" )
elseif( ENABLE_LBMS )
else()
  file( APPEND ${ExprHeader}
    "#include <expression/ExprPatch.h>       // default patch implementation \n"
  )
endif( ENABLE_UINTAH )

file( APPEND ${ExprHeader}
  "#include <expression/Expression.h>      // basic expression support 
#include <expression/ExpressionTree.h>     // support for graphs 
#include <expression/ExpressionFactory.h>  // expression creation help 
#include <expression/TransportEquation.h>  // support for basic transport equations
#include <expression/Functions.h>          // some basic functions wrapped as expressions\n"
)
if( NOT ENABLE_UINTAH )
  file( APPEND ${ExprHeader}
    "#include <expression/TimeStepper.h>        // support for explicit time integrators\n"
    )

  if( ENABLE_OUTPUT )
    file( APPEND ${ExprHeader}
      "#include <expression/FieldWriter.h>        // output support \n"
      )
  endif( ENABLE_OUTPUT )
endif( NOT ENABLE_UINTAH )


# look for git.  This is used to configure version information into the
# executable and also to build upstream dependencies if necessary
set( EXPR_REPO_HASH "\"HASH NOT FOUND\"" )
set( EXPR_REPO_DATE "\"DATE NOT FOUND\"" )
find_package( Git )
if( GIT_FOUND )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%H\""
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE EXPR_REPO_HASH 
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  if( NOT ${RESULT} EQUAL 0 )
    set( EXPR_REPO_HASH "\"HASH NOT FOUND\"" )
  endif()
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%cd\""
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE EXPR_REPO_DATE 
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
   if( NOT ${RESULT} EQUAL 0 )
     set( EXPR_REPO_DATE "\"DATE NOT FOUND\"" )
   endif()
endif( GIT_FOUND )
file( APPEND ${ExprHeader}
 "\n#define EXPR_REPO_DATE ${EXPR_REPO_DATE}  // date of last commit for ExprLib"
 "\n#define EXPR_REPO_HASH ${EXPR_REPO_HASH}  // hash for ExprLib version"
  )


file( APPEND ${ExprHeader} "\n#endif // ExprLib_h\n" )

install( FILES ${ExprHeader} ${FMHeader}
  DESTINATION include/expression
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )
