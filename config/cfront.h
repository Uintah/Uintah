

/*
 * Configuration file for Native compilers
 *
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#define SCI_CCompiler SCI_NativeCCompiler
#define SCI_OptimizeCFlags SCI_NativeOptimizeCFlags
#define SCI_DebugCFlags SCI_NativeDebugCFlags
#define SCI_OtherCFlags SCI_NativeOtherCFlags
#define SCI_CppCompiler SCI_NativeCppCompiler
#define SCI_OptimizeCppFlags SCI_NativeOptimizeCppFlags
#define SCI_DebugCppFlags SCI_NativeDebugCppFlags
#define SCI_OtherCppFlags SCI_NativeOtherCppFlags
#define SCI_CppIncludeLocation SCI_NativeCppIncludeLocation
#define SCI_Linker SCI_NativeLinker
#define SCI_LinkerFlags SCI_NativeLinkerFlags
#ifdef SCI_NativeLinkerNeedsCppFlags
#define SCI_LinkerNeedsCppFlags
#endif
#define SCI_LinkerLib SCI_NativeLinkerLib

#define CppNeedsIncludes
#define LinkerNeedsCppFlags
#define SCI_ShLib CC -G -dy -L/usr/lib -ptr$(TOP)/templates -pti$(TOP) -ptv
#define SCI_CCLibs 

