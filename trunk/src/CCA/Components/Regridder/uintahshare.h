#undef UINTAHSHARE

#ifdef _WIN32
#ifdef BUILD_Packages_Uintah_CCA_Components_Regridder
#define UINTAHSHARE __declspec(dllexport)
#else
#define UINTAHSHARE __declspec(dllimport)
#endif
#else
#define UINTAHSHARE
#endif
