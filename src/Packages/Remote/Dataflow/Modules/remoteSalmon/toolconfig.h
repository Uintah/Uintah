
#ifndef SCI_CONFIG_H
#define SCI_CONFIG_H 1

#ifdef SCI_DEBUG
#define SCI_ASSERTION_LEVEL 2
#else
#define SCI_ASSERTION_LEVEL 0
#endif

#define SCI_USE_JPEG
#define SCI_USE_TIFF

//#ifdef __sgi
#define SCI_MACHINE_sgi
// #define SCI_USE_MP
#define INT64 long long
//#endif

//#ifdef WIN32
//#define SCI_MACHINE_win
//#define SCI_LITTLE_ENDIAN
//#define inline __forceinline
//#define INT64 __int64
//#endif

#endif
