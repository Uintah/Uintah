/* share.h written by Chris Moulding 11/98 */

#undef PSECOMMONSHARE

#ifdef _WIN32
  #if defined(BUILD_PSECOMMON)
    #define PSECOMMONSHARE __declspec(dllexport)
  #else
    #define PSECOMMONSHARE __declspec(dllimport)
  #endif 
#else 
  #define PSECOMMONSHARE 
#endif 
