/* share.h */

#undef CollabVisSHARE

#ifdef _WIN32
  #if defined(BUILD_CollabVis)
    #define CollabVisSHARE __declspec(dllexport)
  #else
    #define CollabVisSHARE __declspec(dllimport)
  #endif
#else
  #define CollabVisSHARE
#endif


