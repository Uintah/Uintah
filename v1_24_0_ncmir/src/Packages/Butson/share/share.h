/* share.h */

#undef ButsonSHARE

#ifdef _WIN32
  #if defined(BUILD_Butson)
    #define ButsonSHARE __declspec(dllexport)
  #else
    #define ButsonSHARE __declspec(dllimport)
  #endif
#else
  #define ButsonSHARE
#endif


