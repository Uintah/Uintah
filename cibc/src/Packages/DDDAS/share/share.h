/* share.h */

#undef DDDASSHARE

#if defined(_WIN32) && !defined(BUILD_DATAFLOW_STATIC)
  #if defined(BUILD_DDDAS)
    #define DDDASSHARE __declspec(dllexport)
  #else
    #define DDDASSHARE __declspec(dllimport)
  #endif
#else
  #define DDDASSHARE
#endif


