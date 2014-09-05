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


#ifndef MODELCREATION_CORE_DATAIO_DATAIOALGO_H
#define MODELCREATION_CORE_DATAIO_DATAIOALGO_H 1

#include <Core/Algorithms/Util/AlgoLibrary.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>
#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>


#include <sgi_stl_warnings_off.h>
#include <string>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class DataIOAlgo : public AlgoLibrary {
public:
  DataIOAlgo(ProgressReporter* pr) : AlgoLibrary(pr) {}

  bool ReadField(std::string filename, FieldHandle& field, std::string importer = "");
  bool WriteField(std::string filename, FieldHandle& field, std::string exporter = "");

  bool ReadMatrix(std::string filename, MatrixHandle& matrix, std::string importer = "");
  bool WriteMatrix(std::string filename, MatrixHandle& matrix, std::string exporter = "");
   
  bool ReadNrrd(std::string filename, NrrdDataHandle& matrix, std::string importer = "");
  bool WriteNrrd(std::string filename, NrrdDataHandle& matrix, std::string exporter = "");

  bool ReadBundle(std::string filename, BundleHandle& matrix, std::string importer = "");
  bool WriteBundle(std::string filename, BundleHandle& matrix, std::string exporter = "");
};

} // end namespace

#endif