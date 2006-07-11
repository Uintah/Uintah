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

#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>

#include <Packages/ModelCreation/Core/Algorithms/TVMHelp.h>

namespace ModelCreation
{

std::string tvm_help_matrix;
std::string tvm_help_field;

extern "C" void * ModelCreationInit(void *param) 
{
  TensorVectorMath::TVMHelp Help;
  tvm_help_matrix = Help.gethelp(false);
  tvm_help_field  = Help.gethelp(true);

  return 0;
}

} // end namespace

namespace SCIRun {
  extern MatrixHandle SimpleTextFileMatrix_reader(ProgressReporter *pr, const char *filename);
  extern bool SimpleTextFileMatrix_writer(ProgressReporter *pr, MatrixHandle matrix, const char *filename);
  extern MatrixHandle NrrdToMatrix_reader(ProgressReporter *pr, const char *filename);
  extern FieldHandle  Nodal_NrrdToField_reader(ProgressReporter *pr, const char *filename);
  extern FieldHandle  Modal_NrrdToField_reader(ProgressReporter *pr, const char *filename);

  static MatrixIEPlugin SimpleTextFileMatrix_plugin("TextFile","", "",SimpleTextFileMatrix_reader,SimpleTextFileMatrix_writer);
  static MatrixIEPlugin NrrdToMatrix_plugin("NrrdFile","{.nhdr} {.nrrd}", "",NrrdToMatrix_reader,0);
  static FieldIEPlugin  NodalNrrdToField_plugin("NrrdFile[DataOnNodes]","{.nhdr} {.nrrd}", "", Nodal_NrrdToField_reader, 0);
  static FieldIEPlugin  ModalNrrdToField_plugin("NrrdFile[DataOnElements]","{.nhdr} {.nrrd}", "", Modal_NrrdToField_reader, 0);
}



