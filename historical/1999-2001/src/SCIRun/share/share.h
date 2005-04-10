/* share.h written by Chris Moulding 11/98 */

#undef SCIRUNSHARE

#ifdef _WIN32
  #if defined(BUILD_SCIRUN)
    #define SCIRUNSHARE __declspec(dllexport)
  #else
    #define SCIRUNSHARE __declspec(dllimport)
  #endif 
#else 
  #define SCIRUNSHARE 
#endif 
