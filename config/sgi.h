

/*
 * Configuration file for SGI
 *
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#define SCI_NativeCCompiler cc
#define SCI_NativeOptimizeCFlags -O2
#define SCI_NativeDebugCFlags -g
#define SCI_NativeOtherCFlags -mips2 -fullwarn
#define SCI_NativeCppCompiler CC
#define SCI_NativeOptimizeCppFlags SCI_NativeOptimizeCFlags
#define SCI_NativeDebugCppFlags SCI_NativeDebugCFlags
#define SCI_NativeOtherCppFlags SCI_OtherCFlags +w -pte.cc -ptr$(TOP)/templates
#define SCI_NativeCppIncludeLocation /usr/include/CC
#define SCI_NativeLinker time CC
#define SCI_NativeLinkerFlags -ptv -pta
#define SCI_NativeLinkerNeedsCppFlags
#define SCI_NativeLinkerLib

#define SCI_DeltaCCompiler cc
#define SCI_DeltaOptimizeCFlags -O
#define SCI_DeltaDebugCFlags -g
#ifdef SCI_IRIX_BIN_o32
#define SCI_BinFlags -32 -mips2
#else
#ifdef SCI_IRIX_BIN_n32_mips3
#define SCI_BinFlags -n32 -mips3
#else
#ifdef SCI_IRIX_BIN_n32_mips4
#define SCI_BinFlags -n32 -mips4
#else
#ifdef SCI_IRIX_BIN_64_mips3
#define SCI_BinFlags -64 -mips3
#else
#ifdef SCI_IRIX_BIN_64_mips4
#define SCI_BinFlags -64 -mips4
#else
#error "Unknown irix binary format"
#endif
#endif
#endif
#endif
#endif

#define SCI_DeltaOtherCFlags -xansi -fullwarn SCI_BinFlags
#define SCI_DeltaCppCompiler CC
#define SCI_DeltaOptimizeCppFlags SCI_DeltaOptimizeCFlags
#define SCI_DeltaDebugCppFlags SCI_DeltaDebugCFlags
#ifdef SCI_IRIX_BIN_32_mips2
#define SCI_BoolFlags -Dbool=int -Dtrue=1 -Dfalse=0
#else
#define SCI_BoolFlags
#endif
#define SCI_DeltaOtherCppFlags SCI_BinFlags SCI_BoolFlags
#define SCI_DeltaCppIncludeLocation /usr/include/CC
#define SCI_DeltaLinker time CC
#define SCI_DeltaLinkerFlags SCI_BinFlags
#define SCI_DeltaLinkerNeedsCppFlags
#define SCI_DeltaLinkerLib

#define SCI_GNUCCompiler gcc
#define SCI_GNUOptimizeCFlags -O2
#define SCI_GNUDebugCFlags -g -O 
#define SCI_GNUOtherCFlags -mcpu-r4000 -mips2 -Wall
#define SCI_GNUCppCompiler g++
#define SCI_GNUCppIncludeLocation /usr/local/gnu/lib/g++-include
#define SCI_GNULinker g++
#define SCI_GNULinkerFlags

#define SCI_Size size

#define SCI_LM -lm
#define SCI_LTHREAD -lmpc
#define SCI_LMOTIF -lXm
#define SCI_LX11 -lX11
#define SCI_LXDR -lsun
#define SCI_LAUDIO -laudio -laudiofile
#define SCI_LGL -lGL -lGLU

#ifdef SCI_PTHREADS
#define SCI_SHLinkTail $(SCI_CLIBS)
#define SCI_ThreadLib -lpthread
#else
#define SCI_SHLinkTail $(SCI_CLIBS)
#define SCI_ThreadLib
#endif

#ifndef SCI_OBJROOT
#define SCI_OBJROOT /home/sci/data1/development/obj/sgi
#endif

