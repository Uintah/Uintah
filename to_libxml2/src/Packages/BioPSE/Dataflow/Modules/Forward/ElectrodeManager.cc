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
 *  ElectrodeManager.cc:
 *
 *  Written by:
 *   lkreda
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>

//#include <Core/Datatypes/TetVolField.h>
//#include <Core/Datatypes/TriSurfField.h>
//#include <Core/Datatypes/PointCloudField.h>
//#include <Dataflow/Modules/Fields/FieldInfo.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
//#include <Dataflow/Widgets/BoxWidget.h>
//#include <Core/Malloc/Allocator.h>
//#include <Core/Math/MinMax.h>
//#include <Core/Math/Trig.h>

#include <Core/GuiInterface/GuiVar.h>
#include <iostream>


namespace BioPSE {

using namespace SCIRun;

class ElectrodeManager : public Module {
  //! Private data

  //! Output port
  MatrixOPort*  electrodeParams_;
  MatrixOPort*  currPattIndicies_;

public:
  GuiInt modelTCL_;
  GuiInt numElTCL_;
  GuiDouble lengthElTCL_;
  GuiInt startNodeIdxTCL_;

  ElectrodeManager(GuiContext*);
  virtual ~ElectrodeManager();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ElectrodeManager)

ElectrodeManager::ElectrodeManager(GuiContext* ctx)
  : Module("ElectrodeManager", ctx, Source, "Forward", "BioPSE"),
    modelTCL_(ctx->subVar("modelTCL")),
    numElTCL_(ctx->subVar("numElTCL")),
    lengthElTCL_(ctx->subVar("lengthElTCL")),
    startNodeIdxTCL_(ctx->subVar("startNodeIdxTCL"))

{
}

ElectrodeManager::~ElectrodeManager(){
}

void
 ElectrodeManager::execute()
{
  electrodeParams_ = (MatrixOPort *)get_oport("Electrode Parameters");
  currPattIndicies_ = (MatrixOPort *)get_oport("Current Pattern Index Vector");

  unsigned int model = modelTCL_.get();
  unsigned int numEl = Max(numElTCL_.get(), 0);
  double lengthEl = Max(lengthElTCL_.get(), 0.0);
  unsigned int startNodeIndex = Max(startNodeIdxTCL_.get(), 0);

  ColumnMatrix* elParams;
  elParams = scinew ColumnMatrix(4);

  if (model==0)
  {
    (*elParams)[0] = 0;
  }
  else
  {
    (*elParams)[0] = 1;  // gap model
  }
  (*elParams)[1]= (double) numEl;
  (*elParams)[2]= lengthEl;
  (*elParams)[3]= startNodeIndex;
 

  // There are numEl-1 unique current patterns
  // Current pattern index is 1-based
  ColumnMatrix* currPattIndicies;
  currPattIndicies = scinew ColumnMatrix(numEl-1);
  for (unsigned int i = 0; i < numEl-1; i++)
  {
    (*currPattIndicies)[i] = i + 1;
  }

  //! Sending result
  electrodeParams_->send(MatrixHandle(elParams)); 
  currPattIndicies_->send(MatrixHandle(currPattIndicies));

}

void
 ElectrodeManager::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace BioPSE


