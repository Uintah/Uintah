/* share.h */

#undef NektarSHARE

#ifdef _WIN32
  #if defined(BUILD_Nektar)
    #define NektarSHARE __declspec(dllexport)
  #else
    #define NektarSHARE __declspec(dllimport)
  #endif 
#else 
  #define NektarSHARE 
#endif 
