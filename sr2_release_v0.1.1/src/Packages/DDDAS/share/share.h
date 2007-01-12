/* share.h */

#undef DDDASSHARE

#ifdef _WIN32
  #if defined(BUILD_DDDAS)
    #define DDDASSHARE __declspec(dllexport)
  #else
    #define DDDASSHARE __declspec(dllimport)
  #endif
#else
  #define DDDASSHARE
#endif


