/* share.h */

#undef RobVSHARE

#ifdef _WIN32
  #if defined(BUILD_RobV)
    #define RobVSHARE __declspec(dllexport)
  #else
    #define RobVSHARE __declspec(dllimport)
  #endif
#else
  #define RobVSHARE
#endif


