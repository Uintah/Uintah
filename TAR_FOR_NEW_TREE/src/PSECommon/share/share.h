/* share.h written by Chris Moulding 11/98 */

#undef PSECommonSHARE

#ifdef _WIN32
  #if defined(BUILD_PSECOMMON)
    #define PSECommonSHARE __declspec(dllexport)
  #else
    #define PSECommonSHARE __declspec(dllimport)
  #endif 
#else 
  #define PSECommonSHARE 
#endif 
