#undef SHARE


#ifdef _WIN32
#ifdef BUILD_Core_Datatypes
#define SHARE __declspec(dllexport)
#else
#define SHARE __declspec(dllimport)
#endif
#else
#define SHARE
#endif
