/* share.h written by Chris Packages/Moulding 11/98 */

#undef Packages/DaveWSHARE

#ifdef _WIN32
  #if defined(BUILD_DAVEW)
    #define Packages/DaveWSHARE __declspec(dllexport)
  #else
    #define Packages/DaveWSHARE __declspec(dllimport)
  #endif 
#else 
  #define Packages/DaveWSHARE 
#endif 
