#undef SCISHARE

#ifdef _WIN32
#ifdef BUILD_Packages_Teem_Dataflow_Modules_Converters
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE __declspec(dllimport)
#endif
#else
#define SCISHARE
#endif
