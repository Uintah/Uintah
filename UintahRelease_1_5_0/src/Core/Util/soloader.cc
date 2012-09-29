/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// soloader.cpp written by Chris Moulding 11/98
#include <Core/Util/Assert.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <iostream>
#include <string>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

using namespace std;

void* GetLibrarySymbolAddress(const char* libname, const char* symbolname)
{
  LIBRARY_HANDLE LibraryHandle = 0;
  
#ifdef _WIN32
  LibraryHandle = LoadLibrary(libname);
  //#elif defined(__APPLE__)
#else
  ASSERT(SCIRun::sci_getenv("SCIRUN_OBJDIR"));
  string name = string(SCIRun::sci_getenv("SCIRUN_OBJDIR")) + "/lib/" + 
    string(libname);
  LibraryHandle = dlopen(name.c_str(), RTLD_LAZY|RTLD_GLOBAL);

  if( LibraryHandle == 0 ) { 
    // dlopen of absolute path failed...  Perhaps they have a DYLD_LIBRARY_PATH var set...
    // If so, if we try again without the path, then maybe it will succeed...
    LibraryHandle = dlopen(libname, RTLD_LAZY|RTLD_GLOBAL);
  }
  //#else
  //LibraryHandle = dlopen(libname, RTLD_LAZY|RTLD_GLOBAL);
#endif
  
  if (LibraryHandle == 0) {
    return 0;
  }
  
#ifdef _WIN32
  return (void*) GetProcAddress(LibraryHandle,symbolname);
#elif defined __APPLE__
  //*** Workaround for a bug in 10.4's dyld library ***//
  // Add a leading underscore to the symbolname for call to mach lib functions
  // If you don't check against the underscored symbol name NSIsSymbolNameDefined
  // will never return true.
  char* underscoredSymbol = 0;
  asprintf(&underscoredSymbol,"_%s",symbolname);
  if( NSIsSymbolNameDefined(underscoredSymbol) ) {
    return dlsym(LibraryHandle,symbolname);
  } else {
    return 0;
  }
#elif defined __APPLE__
  //*** Workaround for a bug in 10.4's dyld library ***//
  // Add a leading underscore to the symbolname for call to mach lib functions
  // If you don't check against the underscored symbol name NSIsSymbolNameDefined
  // will never return true.
  char* underscoredSymbol = 0;
  asprintf(&underscoredSymbol,"_%s",symbolname);
  if( NSIsSymbolNameDefined(underscoredSymbol) ) {
    return dlsym(LibraryHandle,symbolname);
  } else {
    return 0;
  }
#else
  return dlsym(LibraryHandle,symbolname);
#endif
}

LIBRARY_HANDLE
findLib(string lib)
{
  LIBRARY_HANDLE handle=0;
  const char *env = SCIRun::sci_getenv("PACKAGE_LIB_PATH");
  string tempPaths(env?env:"");
  // try to find the library in the specified path
  while (tempPaths!="") {
    string dir;
#ifdef _WIN32
    // make sure we don't throw away the drive letter
    const unsigned int firstColon = tempPaths.find(':',2);
#else
    const unsigned int firstColon = tempPaths.find(':');
#endif
    if(firstColon < tempPaths.size()) {
      dir=tempPaths.substr(0,firstColon);
      tempPaths=tempPaths.substr(firstColon+1);
    } else {
      dir=tempPaths;
      tempPaths="";
    }

    handle = GetLibraryHandle((dir+"/"+lib).c_str());
    if (handle)
      return handle;
  }

  // if not yet found, try to find it in the rpath 
  // or the LD_LIBRARY_PATH (last resort)
  handle = GetLibraryHandle(lib.c_str());
  return handle;
}

void* GetHandleSymbolAddress(LIBRARY_HANDLE handle, const char* symbolname)
{
#ifdef _WIN32
  return (void*) GetProcAddress(handle,symbolname);
#elif defined __APPLE__
  //*** Workaround for a bug in 10.4's dyld library ***//
  // Add a leading underscore to the symbolname for call to mach lib functions
  // If you don't check against the underscored symbol name NSIsSymbolNameDefined
  // will never return true.
  char* underscoredSymbol = 0;
  asprintf(&underscoredSymbol,"_%s",symbolname);
  if( NSIsSymbolNameDefined(underscoredSymbol) ) {
    return dlsym(handle,symbolname);
  } else {
    return 0;
  }
#elif defined __APPLE__
  //*** Workaround for a bug in 10.4's dyld library ***//
  // Add a leading underscore to the symbolname for call to mach lib functions
  // If you don't check against the underscored symbol name NSIsSymbolNameDefined
  // will never return true.
  char* underscoredSymbol = 0;
  asprintf(&underscoredSymbol,"_%s",symbolname);
  if( NSIsSymbolNameDefined(underscoredSymbol) ) {
    return dlsym(handle,symbolname);
  } else {
    return 0;
  }
#else
 return dlsym(handle,symbolname);
#endif
}

LIBRARY_HANDLE GetLibraryHandle(const char* libname)
{
#ifdef _WIN32
  return LoadLibrary(libname);
  //#elif defined(__APPLE__)
#else  
  string name;
  if ( libname[0] == '/' )
    name = libname;
  else {
    ASSERT(SCIRun::sci_getenv("SCIRUN_OBJDIR"));
    name = string(SCIRun::sci_getenv("SCIRUN_OBJDIR"))+"/lib/"+string(libname);
  }

  LIBRARY_HANDLE lh;
  lh = dlopen(name.c_str(), RTLD_LAZY|RTLD_GLOBAL);

  // commented out the following, as it breaks error reporing.  Don't try to 
  // load a second time without dlerror being called and reported.
//   if( lh == 0 ) { 
//     // dlopen of absolute path failed...  Perhaps they have a DYLD_LIBRARY_PATH var set...
//     // If so, if we try again without the path, then maybe it will succeed...
//     lh = dlopen(libname, RTLD_LAZY|RTLD_GLOBAL);
//   }
  return lh;
  //#else
  //return dlopen(libname, RTLD_LAZY|RTLD_GLOBAL);
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

const char* SOError( )
{
#ifdef _WIN32
  char* lpMsgBuf;
  FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
    NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &lpMsgBuf, 0, NULL);

  return lpMsgBuf;
#else
  return dlerror();
#endif
}

LIBRARY_HANDLE FindLibInPath(const std::string& lib, const std::string& path)
{
  LIBRARY_HANDLE handle;
  string tempPaths = path;
  string dir;

  // try to find the library in the specified path
  while (tempPaths!="") {
    const unsigned int firstColon = tempPaths.find(':');
    if(firstColon < tempPaths.size()) {
      dir=tempPaths.substr(0,firstColon);
      tempPaths=tempPaths.substr(firstColon+1);
    } else {
      dir=tempPaths;
      tempPaths="";
    }

    handle = GetLibraryHandle((dir+"/"+lib).c_str());
    if (handle)
      return handle;
  }

  // if not yet found, try to find it in the rpath 
  // or the LD_LIBRARY_PATH (last resort)
  handle = GetLibraryHandle(lib.c_str());
    
  return handle;
}
