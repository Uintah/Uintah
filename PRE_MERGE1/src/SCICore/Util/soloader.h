// soloader.h written by Chris Moulding 11/98
// 
// these functions are used to abstract the interface for 
// accessing shared libraries (.so for unix and .dll for windows)
//

#ifdef _WIN32
#include <windows.h> // for LoadLibrary(), GetProcAddress() and HINSTANCE
typedef HINSTANCE LIBRARY_HANDLE;
#else
#include <dlfcn.h>   // for dlopen() and dlsym()
typedef void* LIBRARY_HANDLE;
#endif

/////////////////////////////////////////////////////////////////
//
// GetLibrarySymbolAddress()
//
// returns a pointer to the data or function called "symbolname"
// from within the shared library called "libname"
//

void* GetLibrarySymbolAddress(char* libname, char* symbolname);


/////////////////////////////////////////////////////////////////
//
// GetLibraryHandle()
//
// opens, and returns the handle to, the library module
// called "libname"
//

LIBRARY_HANDLE GetLibraryHandle(char* libname);


/////////////////////////////////////////////////////////////////
//
// GetHandleSymbolAddress()
//
// returns a pointer to the data or function called "symbolname"
// from within the shared library with handle "handle"
//

void* GetHandleSymbolAddress(LIBRARY_HANDLE handle, char* symbolname);


/////////////////////////////////////////////////////////////////
//
// CloseLibraries()
//
// disassociates all libraries opened, by the calling process, using
// the GetLibrarySymbolAddress() or GetLibraryHandle()
// functions
//

void CloseLibraries();
