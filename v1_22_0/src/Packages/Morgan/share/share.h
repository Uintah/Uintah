/* share.h */

#undef MorganSHARE

#ifdef _WIN32
  #if defined(BUILD_Moulding)
    #define MorganSHARE __declspec(dllexport)
  #else
    #define MorganSHARE __declspec(dllimport)
  #endif
#else
  #define MorganSHARE
#endif


