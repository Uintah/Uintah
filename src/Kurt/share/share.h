/* share.h */

#undef KurtSHARE

#ifdef _WIN32
  #if defined(BUILD_Kurt)
    #define KurtSHARE __declspec(dllexport)
  #else
    #define KurtSHARE __declspec(dllimport)
  #endif 
#else 
  #define KurtSHARE 
#endif 
