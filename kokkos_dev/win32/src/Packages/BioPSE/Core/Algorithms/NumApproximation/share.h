#undef SHARE

#ifdef _WIN32
#ifdef BUILD_Packages_BioPSE_Core_Algorithms_NumApproximation
#define SHARE __declspec(dllexport)
#else
#define SHARE __declspec(dllimport)
#endif
#else
#define SHARE
#endif
