

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
#define SCI_GNUCppIncludeLocation /usr/include/g++ -I/usr/lib/gcc-lib/i486-linux/2.6.2/include
#define SCI_GNUDebugCppFlags -g -O
#define SCI_GNUOtherCppFlags -m486 -Wall -fguiding-decls
#define SCI_GNULinker g++ -rdynamic
#define SCI_GNULinkerFlags
#define SCI_GNULinkerLib

#define SCI_Size size

#define SCI_LM -lm
#define SCI_LTHREAD
#define SCI_LMOTIF -lXt
#define SCI_LX11 -lX11
#define SCI_LXDR
#define SCI_LAUDIO
#define SCI_LGL /home/sparker/direct/glu/libGLU.a /home/sparker/direct/gl/libGL.a /home/sparker/direct/gl/samplegl/libdirectGL.a /home/sparker/direct/gl/libGL.a -lXext

#ifdef SCI_PTHREADS
#define SCI_SHLinkTail $(SCI_CLIBS)
#define SCI_ThreadLib -lpthread
#else
#define SCI_SHLinkTail $(SCI_CLIBS)
#define SCI_ThreadLib
#endif

#define SCI_DYNAMICLINK_LIB -ldl
