/* share.h written by Chris Moulding 11/98 */

#undef PSECORESHARE

#ifdef _WIN32
  #if defined(BUILD_PSECORE)
    #define PSECORESHARE __declspec(dllexport)
  #else
    #define PSECORESHARE __declspec(dllimport)
  #endif 
#else 
  #define PSECORESHARE 
#endif 
