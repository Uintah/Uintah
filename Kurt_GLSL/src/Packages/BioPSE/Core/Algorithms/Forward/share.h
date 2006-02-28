#undef SCISHARE

#ifdef _WIN32
#ifdef BUILD_Packages_BioPSE_Core_Algorithms_Forward
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE __declspec(dllimport)
#endif
#else
#define SCISHARE
#endif
