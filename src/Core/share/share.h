/* share.h written by Chris Moulding 11/98 */

#undef SCICORESHARE

#ifdef _WIN32
  #if defined(BUILD_SCICORE)
    #define SCICORESHARE __declspec(dllexport)
  #else
    #define SCICORESHARE __declspec(dllimport)
  #endif 
#else 
  #define SCICORESHARE 
#endif 
