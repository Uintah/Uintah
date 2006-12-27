/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2005 Scientific Computing and Imaging Institute,
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


/*
 *  init.cc: 
 *
 *  Written by:
 *   McKay Davis
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   Mar 2005
 *
 *  Copyright (C) 2005 U of U
 */

#include <Core/Init/init.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/Containers/StringUtil.h>
//#include <Dataflow/Network/PackageDB.h>

using namespace SCIRun;
using std::string;

#ifdef __APPLE__
  static string lib_ext = ".dylib";
#elif defined(_WIN32)
  const string lib_ext = ".dll";
#else
  static string lib_ext = ".so";
#endif


// SCIRunInit is called from from main() (and from stantalone converters).
//
// This method calls ${PACKAGE}Init(0) for every package in the comma-seperated
// 'packages' string.  Called for every package in SCIRUN_LOAD_PACKAGE
// if packages is empty (which is the default)
//
// Note, the void * ${PACKAGE}Init(void *) function must be declared extern C
// in the SCIRun/src/Packages/${PACKAGE}/Core/Datatypes directory
//
// For example, for the BioPSE package:
//
// extern "C" void * BioPSEInit(void *param) 
// {
//   std::cerr << "BioPSEInit called.\n";
//   return 0;
// }


void
SCIRunInit(string packages) 
{
  string package_list = packages;

  if( package_list.empty() ) {
    const char * env_value = sci_getenv("SCIRUN_LOAD_PACKAGE");
    if( env_value ) {
      package_list = env_value;
    }
  }

  vector <string> package = split_string(package_list, ',');
  typedef void *(*PackageInitFunc)(void *);

  for (unsigned int i = 0; i < package.size(); ++i) {
    LIBRARY_HANDLE lib = findLib("libPackages_"+package[i]+"_Core_Datatypes"+lib_ext);
    PackageInitFunc init = 0;
    if (lib)
      init = (PackageInitFunc)GetHandleSymbolAddress
	(lib, (package[i]+"Init").c_str());
    if (init) {
      (*init)(0);
    }
  }

}

