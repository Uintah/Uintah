/* share.h */

#undef YardenSHARE

#ifdef _WIN32
  #if defined(BUILD_Yarden)
    #define YardenSHARE __declspec(dllexport)
  #else
    #define YardenSHARE __declspec(dllimport)
  #endif 
#else 
  #define YardenSHARE 
#endif 
