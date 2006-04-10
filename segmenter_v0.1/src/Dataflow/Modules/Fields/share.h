#undef SCISHARE

#ifdef _WIN32
#ifdef BUILD_Dataflow_Modules_Fields
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE __declspec(dllimport)
#endif
#else
#define SCISHARE
#endif
