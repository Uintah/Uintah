/* share.h */

#undef Packages/NektarSHARE

#ifdef _WIN32
  #if defined(BUILD_Packages/Nektar)
    #define Packages/NektarSHARE __declspec(dllexport)
  #else
    #define Packages/NektarSHARE __declspec(dllimport)
  #endif 
#else 
  #define Packages/NektarSHARE 
#endif 
