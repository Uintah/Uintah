/* share.h */

#undef Packages/MouldingSHARE

#ifdef _WIN32
  #if defined(BUILD_Packages/Moulding)
    #define Packages/MouldingSHARE __declspec(dllexport)
  #else
    #define Packages/MouldingSHARE __declspec(dllimport)
  #endif
#else
  #define Packages/MouldingSHARE
#endif


