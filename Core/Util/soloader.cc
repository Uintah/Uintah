// soloader.cpp written by Chris Moulding 11/98

#include <iostream>
using std::cerr;
using std::endl;
#include <Core/Util/soloader.h>

void* GetLibrarySymbolAddress(const char* libname, const char* symbolname)
{
  LIBRARY_HANDLE LibraryHandle = 0;
  
#ifdef _WIN32
  LibraryHandle = LoadLibrary(libname);
#else
  LibraryHandle = dlopen(libname, RTLD_LAZY);
#endif
  
  if (LibraryHandle == 0) 
    return 0;
  
#ifdef _WIN32
  return GetProcAddress(LibraryHandle,symbolname);
#else
  return dlsym(LibraryHandle,symbolname);
#endif
}

void* GetHandleSymbolAddress(LIBRARY_HANDLE handle, const char* symbolname)
{
#ifdef _WIN32
  return GetProcAddress(handle,symbolname);
#else
  return dlsym(handle,symbolname);
#endif
}

LIBRARY_HANDLE GetLibraryHandle(const char* libname)
{
#ifdef _WIN32
  return LoadLibrary(libname);
#else
  return dlopen(libname, RTLD_LAZY);
#endif
}

void CloseLibrary(LIBRARY_HANDLE LibraryHandle)
{
#ifdef _WIN32
  FreeLibrary(LibraryHandle);
#else
  dlclose(LibraryHandle);
#endif
}

const char* SOError()
{
#ifdef _WIN32
  return 0;
#else
  return dlerror();
#endif
}

