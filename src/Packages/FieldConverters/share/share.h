/* share.h */

#undef FieldConvertersSHARE

#ifdef _WIN32
  #if defined(BUILD_FieldConverters)
    #define FieldConvertersSHARE __declspec(dllexport)
  #else
    #define FieldConvertersSHARE __declspec(dllimport)
  #endif
#else
  #define FieldConvertersSHARE
#endif


