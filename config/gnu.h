
/*
 * Configuration file for GNU compilers
 *
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#define SCI_CCompiler SCI_GNUCCompiler
#define SCI_OptimizeCFlags SCI_GNUOptimizeCFlags
#define SCI_DebugCFlags SCI_GNUDebugCFlags
#define SCI_OtherCFlags SCI_GNUOtherCFlags
#define SCI_CppCompiler SCI_GNUCppCompiler
#define SCI_OptimizeCppFlags SCI_GNUOptimizeCppFlags
#define SCI_DebugCppFlags SCI_GNUDebugCppFlags
#define SCI_OtherCppFlags SCI_GNUOtherCppFlags
#define SCI_CppIncludeLocation SCI_GNUCppIncludeLocation
#define SCI_Linker SCI_GNULinker
#define SCI_LinkerFlags SCI_GNULinkerFlags
#ifdef SCI_GNULinkerNeedsCppFlags
#define SCI_LinkerNeedsCppFlags
#endif
#define SCI_LinkerLib SCI_GNULinkerLib
#define SCI_ShLib g++ -shared
#define SCI_CCLibs

#define SCI_TemplateLib
