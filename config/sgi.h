

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
#ifdef SCI_IRIX6
#ifdef SCI_64BIT
#define SCI_DeltaOtherCFlags -mips4 -fullwarn
#else
#ifdef SCI_N32
#define SCI_DeltaOtherCFlags -n32 -mips4 -fullwarn
#else
#define SCI_DeltaOtherCFlags -32 -mips2 -fullwarn
#endif
#endif
#else
#define SCI_DeltaOtherCFlags -mips2 -fullwarn
#endif

#define SCI_DeltaCppCompiler CC
#define SCI_DeltaOptimizeCppFlags SCI_DeltaOptimizeCFlags
#define SCI_DeltaDebugCppFlags SCI_DeltaDebugCFlags
#ifdef SCI_IRIX6
#ifdef SCI_64BIT
#define SCI_DeltaOtherCppFlags -mips4
#else
#ifdef SCI_N32
#define SCI_DeltaOtherCppFlags -n32 -mips4
#else
#define SCI_DeltaOtherCppFlags -32 -mips2
#endif
#endif
#else
#define SCI_DeltaOtherCppFlags -mips2
#endif
#define SCI_DeltaCppIncludeLocation /usr/include/CC
#define SCI_DeltaLinker time CC
#ifdef SCI_64BIT
#define SCI_DeltaLinkerFlags -64
#else
#ifdef SCI_N32
#define SCI_DeltaLinkerFlags -n32
#else
#define SCI_DeltaLinkerFlags -32
#endif
#endif
#define SCI_DeltaLinkerNeedsCppFlags
#define SCI_DeltaLinkerLib
#if 0
#define SCI_DeltaPrelinkCommand /usr/lib/DCC/edg_prelink -v
#endif

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

#ifndef SCI_OBJROOT
#define SCI_OBJROOT /home/sci/data1/development/obj/sgi
#endif

