/* share.h */

#undef Packages/BioPSESHARE

#ifdef _WIN32
  #if defined(BUILD_Packages/BioPSE)
    #define Packages/BioPSESHARE __declspec(dllexport)
  #else
    #define Packages/BioPSESHARE __declspec(dllimport)
  #endif
#else
  #define Packages/BioPSESHARE
#endif


