/* share.h */

#undef BioPSESHARE

#ifdef _WIN32
  #if defined(BUILD_Nrrd)
    #define BioPSESHARE __declspec(dllexport)
  #else
    #define BioPSESHARE __declspec(dllimport)
  #endif
#else
  #define BioPSESHARE
#endif


