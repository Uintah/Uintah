/* share.h */

#undef Packages/YardenSHARE

#ifdef _WIN32
  #if defined(BUILD_Packages/Yarden)
    #define Packages/YardenSHARE __declspec(dllexport)
  #else
    #define Packages/YardenSHARE __declspec(dllimport)
  #endif 
#else 
  #define Packages/YardenSHARE 
#endif 
