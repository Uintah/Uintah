/* share.h written by Chris Moulding 11/98 */

#undef UINTAHSHARE

#ifdef _WIN32
  #if defined(BUILD_UINTAH)
    #define UINTAHSHARE __declspec(dllexport)
  #else
    #define UINTAHSHARE __declspec(dllimport)
  #endif 
#else 
  #define UINTAHSHARE 
#endif 
