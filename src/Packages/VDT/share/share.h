/* share.h */

#undef VDTSHARE

#ifdef _WIN32
  #if defined(BUILD_VDT)
    #define VDTSHARE __declspec(dllexport)
  #else
    #define VDTSHARE __declspec(dllimport)
  #endif
#else
  #define VDTSHARE
#endif


