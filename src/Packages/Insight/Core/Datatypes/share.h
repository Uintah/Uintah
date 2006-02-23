#undef SCISHARE

#ifdef _WIN32
#ifdef BUILD_Packages_Insight_Core_Datatypes
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE __declspec(dllimport)
#endif
#else
#define SCISHARE
#endif
