/* share.h */

#undef Packages/KurtSHARE

#ifdef _WIN32
  #if defined(BUILD_Packages/Kurt)
    #define Packages/KurtSHARE __declspec(dllexport)
  #else
    #define Packages/KurtSHARE __declspec(dllimport)
  #endif 
#else 
  #define Packages/KurtSHARE 
#endif 
