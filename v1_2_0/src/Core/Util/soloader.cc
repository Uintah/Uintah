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

