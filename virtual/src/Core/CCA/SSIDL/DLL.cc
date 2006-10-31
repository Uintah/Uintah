/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#include <Core/CCA/SSIDL/sidl_sidl.h>
#include <Core/Util/NotFinished.h>

using SSIDL::DLL;
using SSIDL::BaseClass;

// bool .SSIDL.DLL.loadLibrary(in string uri, in bool loadGlobally, in bool loadLazy)
bool DLL::loadLibrary(const std::string& uri, bool loadGlobally, bool loadLazy)
{
  NOT_FINISHED("bool .SSIDL.DLL.loadLibrary(in string uri, in bool loadGlobally, in bool loadLazy)");
  return false;
}

// string .SSIDL.DLL.getName()
std::string DLL::getName()
{
  NOT_FINISHED("string .SSIDL.DLL.getName()");
  return false;
}

// bool .SSIDL.DLL.isGlobal()
bool DLL::isGlobal()
{
  NOT_FINISHED("bool .SSIDL.DLL.isGlobal()");
  return false;
}

// bool .SSIDL.DLL.isLazy()
bool DLL::isLazy()
{
  NOT_FINISHED("bool .SSIDL.DLL.isLazy()");
  return false;
}

// void .SSIDL.DLL.unloadLibrary()
void DLL::unloadLibrary()
{
  NOT_FINISHED("void .SSIDL.DLL.unloadLibrary()");
}

// void* .SSIDL.DLL.lookupSymbol(in string linker_name)
void* DLL::lookupSymbol(const std::string& linker_name)
{
  NOT_FINISHED("void* .SSIDL.DLL.lookupSymbol(in string linker_name)");
  return 0;
}

// .SSIDL.BaseClass .SSIDL.DLL.createClass(in string sidl_name)
BaseClass::pointer DLL::createClass(const std::string& sidl_name)
{
  NOT_FINISHED(".SSIDL.BaseClass .SSIDL.DLL.createClass(in string sidl_name)");
  return BaseClass::pointer(0);
}
