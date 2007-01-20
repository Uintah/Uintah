#undef SCISHARE

#if defined(_WIN32) && !defined(BUILD_CORE_STATIC)
#ifdef BUILD_Core_Algorithms_Fields
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE __declspec(dllimport)
#endif
#else
#define SCISHARE
#endif
