/* share.h */

#undef Packages/ButsonSHARE

#ifdef _WIN32
  #if defined(BUILD_Packages/Butson)
    #define Packages/ButsonSHARE __declspec(dllexport)
  #else
    #define Packages/ButsonSHARE __declspec(dllimport)
  #endif
#else
  #define Packages/ButsonSHARE
#endif


