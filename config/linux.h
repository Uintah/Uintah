

/*
 * Configuration file for Linux
 *
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#define SCI_GNUCCompiler gcc
#define SCI_GNUOptimizeCFlags -O2
#define SCI_GNUDebugCFlags -g -O 
#define SCI_GNUOtherCFlags -m486 -Wall
#define SCI_GNUCppCompiler g++
#define SCI_GNUCppIncludeLocation /usr/g++-include
#define SCI_GNUDebugCppFlags -g -O
#define SCI_GNUOtherCppFlags -m486 -Wall -fexternal-templates
#define SCI_GNULinker g++
#define SCI_GNULinkerFlags
#define SCI_GNULinkerLib

#define SCI_Size size

#define SCI_LM -lm
#define SCI_LTHREAD
#define SCI_LMOTIF -lXt
#define SCI_LX11 -lX11
#define SCI_LXDR
#define SCI_LAUDIO
#define SCI_LGL
