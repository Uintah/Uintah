/* share.h */

#undef MITSHARE

#ifdef _WIN32
  #if defined(BUILD_MIT)
    #define MITSHARE __declspec(dllexport)
  #else
    #define MITSHARE __declspec(dllimport)
  #endif
#else
  #define MITSHARE
#endif


