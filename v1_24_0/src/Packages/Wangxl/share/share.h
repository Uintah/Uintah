/* share.h */

#undef WangxlSHARE

#ifdef _WIN32
  #if defined(BUILD_Wangxl)
    #define WangxlSHARE __declspec(dllexport)
  #else
    #define WangxlSHARE __declspec(dllimport)
  #endif
#else
  #define WangxlSHARE
#endif


