/* share.h written by Chris Packages/Moulding 11/98 */

#undef DAVEWSHARE

#ifdef _WIN32
  #if defined(BUILD_DAVEW)
    #define DAVEWSHARE __declspec(dllexport)
  #else
    #define DAVEWSHARE __declspec(dllimport)
  #endif 
#else 
  #define DAVEWSHARE 
#endif 
