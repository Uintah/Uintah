/* share.h */

#undef NrrdSHARE

#ifdef _WIN32
  #if defined(BUILD_Nrrd)
    #define NrrdSHARE __declspec(dllexport)
  #else
    #define NrrdSHARE __declspec(dllimport)
  #endif
#else
  #define NrrdSHARE
#endif
