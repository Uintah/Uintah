#undef SCISHARE

#if defined(_WIN32) && !defined(BUILD_DATAFLOW_STATIC)
#ifdef BUILD_Packages_BioPSE_Core_Datatypes
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE __declspec(dllimport)
#endif
#else
#define SCISHARE
#endif

