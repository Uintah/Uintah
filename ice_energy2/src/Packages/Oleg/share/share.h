/* share.h */

#undef OlegSHARE

#ifdef _WIN32
  #if defined(BUILD_Oleg)
    #define OlegSHARE __declspec(dllexport)
  #else
    #define OlegSHARE __declspec(dllimport)
  #endif
#else
  #define OlegSHARE
#endif


