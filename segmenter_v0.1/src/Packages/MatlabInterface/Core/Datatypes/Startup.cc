/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
 *  Module.h: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <string>
#include <iostream>
#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#include <Core/ImportExport/Nrrd/NrrdIEPlugin.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabconverter.h>

using namespace std;
using namespace SCIRun;

extern MatrixHandle MatlabMatrix_reader(ProgressReporter *pr, const char *filename);
extern bool MatlabMatrix_writer(ProgressReporter *pr, MatrixHandle mh, const char *filename);
extern FieldHandle MatlabField_reader(ProgressReporter *pr, const char *filename);
extern bool MatlabField_writer(ProgressReporter *pr, FieldHandle mh, const char *filename);
extern NrrdDataHandle MatlabNrrd_reader(ProgressReporter *pr, const char *filename);
extern bool MatlabNrrd_writer(ProgressReporter *pr, NrrdDataHandle mh, const char *filename);

namespace MatlabIO 
{

// It is enough that this part of the library is called to force
// the Macintosh to initialize the modules

extern "C" void * MatlabInterfaceInit(void *param) 
{
  return 0;
}

static MatrixIEPlugin MatlabMatrix_plugin("Matlab Matrix",".mat", "*.mat", MatlabMatrix_reader, MatlabMatrix_writer);
static FieldIEPlugin MatlabField_plugin("Matlab Field",".mat", "*.mat",MatlabField_reader,MatlabField_writer);   
static NrrdIEPlugin MatlabNrrd_plugin("Matlab Matrix",".mat", "*.mat",MatlabNrrd_reader,MatlabNrrd_writer);

}

