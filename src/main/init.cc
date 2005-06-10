/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2005 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
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

#include <main/init.h>
#include <Dataflow/Network/PackageDB.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/Containers/StringUtil.h>

using namespace SCIRun;
using std::string;

#ifdef __APPLE__
static string lib_ext = ".dylib";
#elif defined(_WIN32)
const string lib_ext = ".dll";
#else
static string lib_ext = ".so";
#endif


#if defined(__APPLE__)  

namespace SCIRun {
extern void macImportExportForceLoad();
}


// #include <Core/Datatypes/PointCloudField.h>
// #include <Core/Datatypes/TriSurfField.h>
// #include <Core/Datatypes/StructQuadSurfField.h>
// #include <Core/Datatypes/TriSurfMesh.h>
// #include <Core/Datatypes/StructHexVolField.h>
#include <iostream>
 
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Basis/HexTricubicHmtScaleFactors.h>
#include <Core/Datatypes/HexVolMesh.h>
//  The Macintosh does not run static object constructors when loading a 
//  dynamic library until a symbol from that library is referenced.  See:
//
//  http://wwwlapp.in2p3.fr/~buskulic/static_constructors_in_Darwin.html
//
//  We use this function to force Core/Datatypes to load and run constructors
//  (but only for Macs.)
void
macForceLoad()
{
  std::cout << "Forcing load of Core/Datatypes (for Macintosh)\n";

  // It appears that we need more than one type to force the
  // instantiation of all of Core/Datatypes.  Until we find a better
  // solution (or upgrade to the next OS version (jaguar) which I
  // think will fix this) I suggest that we just add the types to this
  // file as we find them "missing"...  -Dd

//   PointCloudField<double> pcfd;
//   TriSurfField<double> tsfd;
//   StructQuadSurfField<double> sqsfd;
//   TriSurfMesh tsmd;
//   StructHexVolField<double> shvfd;

typedef HexVolMesh<HexTricubicHmtScaleFactors<Point> > HVMesh;
typedef HexTricubicHmtScaleFactors<double> DatBasis;
typedef GenericField<HVMesh, DatBasis, vector<double> > HVField;
 HVField dummy;

//   PointCloudField<double> pcfd;
//   TriSurfField<double> tsfd;
//   StructQuadSurfField<double> sqsfd;
//   TriSurfMesh tsmd;
//   StructHexVolField<double> shvfd;
>>>>>>> .r30657
}
#endif



// SCIRunInit is called from from main()
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
//{
//  std::cerr << "BioPSEInit called.\n";
//  return 0;
//}



void
SCIRunInit(string packages) {
#if defined(__APPLE__)  
  macImportExportForceLoad(); // Attempting to force load (and thus
                              // instantiation of static constructors) 
  macForceLoad();             // of Core/Datatypes and Core/ImportExport.
#endif

  if (packages.empty())
    packages = sci_getenv("SCIRUN_LOAD_PACKAGE");
  vector <string> package = split_string(packages, ',');
  typedef void *(*PackageInitFunc)(void *);

  for (unsigned int i = 0; i < package.size(); ++i) {
    LIBRARY_HANDLE lib = 
      PackageDB::findLib("libPackages_"+package[i]+"_Core_Datatypes"+lib_ext);
    PackageInitFunc init = 0;
    if (lib)
      init = (PackageInitFunc)GetHandleSymbolAddress
	(lib, (package[i]+"Init").c_str());
    if (init) {
      (*init)(0);
    }
  }
}

