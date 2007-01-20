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


#ifndef PACKAGES_CARDIOWAVE_CORE_MODEL_MODELALGO_H
#define PACKAGES_CARDIOWAVE_CORE_MODEL_MODELALGO_H 1

#include <Core/Util/AlgoLibrary.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Network/Module.h>
#include <Packages/CardioWaveInterface/Core/Model/BuildMembraneTable.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace CardioWaveInterface {

using namespace SCIRun;
using namespace ModelCreation;

class ModelAlgo : public AlgoLibrary {

  public:
    ModelAlgo(ProgressReporter* pr); // normal case

    bool DMDBuildDomain(FieldHandle elementtype, FieldHandle conductivity, std::vector<FieldHandle> membrane, MatrixHandle& femmatrix, MatrixHandle& volvec, MatrixHandle& nodetype);    
    bool DMDBuildSimulator(BundleHandle Model, std::string filename);
    bool DMDBuildMembraneTable(FieldHandle elementtype, FieldHandle membranemodel, MembraneTableList& table);


};

}

#endif
