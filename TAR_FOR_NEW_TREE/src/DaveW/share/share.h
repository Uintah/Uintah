/* share.h written by Chris Moulding 11/98 */

#undef DaveWSHARE

#ifdef _WIN32
  #if defined(BUILD_DAVEW)
    #define DaveWSHARE __declspec(dllexport)
  #else
    #define DaveWSHARE __declspec(dllimport)
  #endif 
#else 
  #define DaveWSHARE 
#endif 
