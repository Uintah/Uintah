/* share.h */

#undef NetSolveSHARE

#ifdef _WIN32
  #if defined(BUILD_NetSolve)
    #define NetSolveSHARE __declspec(dllexport)
  #else
    #define NetSolveSHARE __declspec(dllimport)
  #endif
#else
  #define NetSolveSHARE
#endif


