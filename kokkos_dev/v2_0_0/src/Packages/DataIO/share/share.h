/* share.h */

#undef DataIOSHARE

#ifdef _WIN32
  #if defined(BUILD_DataIO)
    #define DataIOSHARE __declspec(dllexport)
  #else
    #define DataIOSHARE __declspec(dllimport)
  #endif
#else
  #define DataIOSHARE
#endif


