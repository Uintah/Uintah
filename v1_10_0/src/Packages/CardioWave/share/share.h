/* share.h */

#undef CardioWaveSHARE

#ifdef _WIN32
  #if defined(BUILD_CardioWave)
    #define CardioWaveSHARE __declspec(dllexport)
  #else
    #define CardioWaveSHARE __declspec(dllimport)
  #endif
#else
  #define CardioWaveSHARE
#endif


