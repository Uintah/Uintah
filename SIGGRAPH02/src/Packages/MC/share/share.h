/* share.h */

#undef MCSHARE

#ifdef _WIN32
  #if defined(BUILD_MC)
    #define MCSHARE __declspec(dllexport)
  #else
    #define MCSHARE __declspec(dllimport)
  #endif
#else
  #define MCSHARE
#endif


