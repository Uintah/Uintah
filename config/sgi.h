

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
#define SCI_NativeLinkerFlags -ptv
#define SCI_NativeLinkerNeedsCppFlags
#define SCI_NativeLinkerLib

#define SCI_GNUCCompiler gcc
#define SCI_GNUOptimizeCFlags -O2
#define SCI_GNUDebugCFlags -g -O 
#define SCI_GNUOtherCFlags -mcpu-r4000 -mips2 -Wall
#define SCI_GNUCppCompiler g++
#define SCI_GNUCppIncludeLocation /usr/local/gnu/lib/g++-include
#define SCI_GNULinker g++
#define SCI_GNULinkerFlags

#define SCI_TemplateLib $(TOP)/templates/fast_templates.o

#define SCI_Size size

#define SCI_LM -lm
#define SCI_LTHREAD -lmpc
#define SCI_LMOTIF -lXm
#define SCI_LX11 -lX11
#define SCI_LXDR -lsun
#define SCI_LAUDIO -laudio -laudiofile
#define SCI_LGL -lGL

#ifndef SCI_OBJROOT
#define SCI_OBJROOT /home/sci/data1/development/obj/sgi
#endif

