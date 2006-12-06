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

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Core/GuiInterface/GuiVar.h>
#include <iostream>


namespace BioPSE {

using namespace SCIRun;

class ElectrodeManager : public Module {

public:
  GuiInt modelTCL_;
  GuiInt numElTCL_;
  GuiDouble lengthElTCL_;
  GuiInt startNodeIdxTCL_;

  ElectrodeManager(GuiContext*);
  virtual ~ElectrodeManager();

  virtual void execute();
};


DECLARE_MAKER(ElectrodeManager)

ElectrodeManager::ElectrodeManager(GuiContext* ctx)
  : Module("ElectrodeManager", ctx, Source, "Forward", "BioPSE"),
    modelTCL_(get_ctx()->subVar("modelTCL")),
    numElTCL_(get_ctx()->subVar("numElTCL")),
    lengthElTCL_(get_ctx()->subVar("lengthElTCL")),
    startNodeIdxTCL_(get_ctx()->subVar("startNodeIdxTCL"))

{
}


ElectrodeManager::~ElectrodeManager()
{
}


void
ElectrodeManager::execute()
{
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
  MatrixHandle ehandle(elParams);
  send_output_handle("Electrode Parameters", ehandle);

  MatrixHandle chandle(currPattIndicies);
  send_output_handle("Current Pattern Index Vector", chandle);
}


} // End namespace BioPSE


