/* share.h */

#undef MouldingSHARE

#ifdef _WIN32
  #if defined(BUILD_Moulding)
    #define MouldingSHARE __declspec(dllexport)
  #else
    #define MouldingSHARE __declspec(dllimport)
  #endif
#else
  #define MouldingSHARE
#endif


