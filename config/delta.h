

/*
 * Configuration file for Delta compilers
 *
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#define SCI_CCompiler SCI_DeltaCCompiler
#define SCI_OptimizeCFlags SCI_DeltaOptimizeCFlags
#define SCI_DebugCFlags SCI_DeltaDebugCFlags
#define SCI_OtherCFlags SCI_DeltaOtherCFlags
#define SCI_CppCompiler SCI_DeltaCppCompiler
#define SCI_OptimizeCppFlags SCI_DeltaOptimizeCppFlags
#define SCI_DebugCppFlags SCI_DeltaDebugCppFlags
#define SCI_OtherCppFlags SCI_DeltaOtherCppFlags
#define SCI_CppIncludeLocation SCI_DeltaCppIncludeLocation
#define SCI_Linker SCI_DeltaLinker
#define SCI_LinkerFlags SCI_DeltaLinkerFlags
#define SCI_LinkerLib SCI_DeltaLinkerLib

#define SCI_ShLib CC SCI_BinFlags -shared -Wl,-no_unresolved -update_registry $(TOP)/so_locations
#define SCI_CCLibs -lC -lc

#define CppNeedsIncludes
#define LinkerNeedsCppFlags
#define SCI_TemplateLib
