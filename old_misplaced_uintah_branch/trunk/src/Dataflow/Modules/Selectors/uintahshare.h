#undef UINTAHSHARE

#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#ifdef BUILD_Packages_Uintah_Dataflow_Modules_Selectors
#define UINTAHSHARE __declspec(dllexport)
#else
#define UINTAHSHARE __declspec(dllimport)
#endif
#else
#define UINTAHSHARE
#endif
