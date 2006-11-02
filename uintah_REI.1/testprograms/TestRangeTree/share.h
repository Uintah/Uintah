#undef SCISHARE

#ifdef _WIN32
#ifdef BUILD_Packages_Uintah_testprograms_TestRangeTree
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE __declspec(dllimport)
#endif
#else
#define SCISHARE
#endif
