/* share.h */

#undef Packages/RobVSHARE

#ifdef _WIN32
  #if defined(BUILD_Packages/RobV)
    #define Packages/RobVSHARE __declspec(dllexport)
  #else
    #define Packages/RobVSHARE __declspec(dllimport)
  #endif
#else
  #define Packages/RobVSHARE
#endif


