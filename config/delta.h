

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
#if 0
#define SCI_PrelinkCommand SCI_DeltaPrelinkCommand
#endif
#ifdef SCI_64BIT
#define SCI_ShLib CC -64 -shared -Wl,-no_unresolved
#else
#ifdef SCI_N32
#define SCI_ShLib CC -n32 -shared -Wl,-no_unresolved
#else
#define SCI_ShLib CC -32 -shared -Wl,-no_unresolved
#endif
#endif
#define SCI_CCLibs -lC -lc

#define CppNeedsIncludes
#define LinkerNeedsCppFlags
#define SCI_TemplateLib
