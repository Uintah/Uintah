/* share.h */

#undef FusionSHARE

#ifdef _WIN32
  #if defined(BUILD_Fusion)
    #define FusionSHARE __declspec(dllexport)
  #else
    #define FusionSHARE __declspec(dllimport)
  #endif
#else
  #define FusionSHARE
#endif


