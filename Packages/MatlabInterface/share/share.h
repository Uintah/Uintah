/* share.h */

#undef MatlabInterfaceSHARE

#ifdef _WIN32
  #if defined(BUILD_MatlabInterface)
    #define MatlabInterfaceSHARE __declspec(dllexport)
  #else
    #define MatlabInterfaceSHARE __declspec(dllimport)
  #endif
#else
  #define MatlabInterfaceSHARE
#endif


