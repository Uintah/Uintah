/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

// soloader.cpp written by Chris Moulding 11/98

#include <Core/Util/soloader.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>
using namespace std;

void* GetLibrarySymbolAddress(const char* libname, const char* symbolname)
{
  LIBRARY_HANDLE LibraryHandle = 0;
  
#ifdef _WIN32
  LibraryHandle = LoadLibrary(libname);
#elif defined(__APPLE__)
  string name = string("lib/")+libname;
  LibraryHandle = dlopen(name.c_str(), RTLD_LAZY|RTLD_GLOBAL);
#else
  LibraryHandle = dlopen(libname, RTLD_LAZY|RTLD_GLOBAL);
#endif
  
  if (LibraryHandle == 0) {
    return 0;
  }
  
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
#elif defined(__APPLE__) && __GNUC__ == 3 && __GNUC_MINOR == 1 
  string name("_");
  name += symbolname;
  return dlsym(handle, name.c_str());
#else
 return dlsym(handle,symbolname);
#endif
}

LIBRARY_HANDLE GetLibraryHandle(const char* libname)
{
#ifdef _WIN32
  return LoadLibrary(libname);
#elif defined(__APPLE__)
  string name = string("lib/") + libname;
  return dlopen(name.c_str(), RTLD_LAZY|RTLD_GLOBAL);
#else
  return dlopen(libname, RTLD_LAZY|RTLD_GLOBAL);
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
  return 0;
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
