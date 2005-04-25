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
 *  TrigCurrentPattern.cc: Generates a trigonometric current pattern for EIT testing.
 *  
 *  Written by:
 *   lkreda
 *   November 30, 2003
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class TrigCurrentPattern : public Module {
public:
  GuiDouble magnitudeTCL_;

  TrigCurrentPattern(GuiContext*);
  virtual ~TrigCurrentPattern();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};

DECLARE_MAKER(TrigCurrentPattern)
TrigCurrentPattern::TrigCurrentPattern(GuiContext* ctx)
  : Module("TrigCurrentPattern", ctx, Source, "Forward", "BioPSE")
  , magnitudeTCL_(ctx->subVar("magnitudeTCL"))
{
}

TrigCurrentPattern::~TrigCurrentPattern(){
}

void
 TrigCurrentPattern::execute(){
 
  cout << "In TrigCurrentPattern " << endl;

  //! Input ports
  MatrixIPort*  iportCurrentPatternIndex_;
  MatrixIPort*  iportElectrodeParams_;

  //! Output ports
  MatrixOPort*  oportCurrentVector_;

  iportCurrentPatternIndex_ = (MatrixIPort *)get_iport("CurrentPatternIndex");
  iportElectrodeParams_ = (MatrixIPort *)get_iport("Electrode Parameters");

  oportCurrentVector_ = (MatrixOPort *)get_oport("CurrentPatternVector");

  double currentMagnitude = Max(magnitudeTCL_.get(), 0.0);


  cout << "Current magnitude = " << currentMagnitude << endl;

  // Get the current pattern index  
  MatrixHandle  hCurrentPatternIndex;
  ColumnMatrix* currPatIdx;
  int           k;

  // -- copy the input current pattern index into local variable, k 

  if (iportCurrentPatternIndex_->get(hCurrentPatternIndex) && 
      (currPatIdx=dynamic_cast<ColumnMatrix*>(hCurrentPatternIndex.get_rep())) && 
      (currPatIdx->nrows() == 1))
  {
    k=static_cast<int>((*currPatIdx)[0]);
  }
  else{
    msgStream_ << "The supplied current pattern index is not a 1x1 matrix" << endl;
  }

  cout << "Supplied current pattern index is " << k << endl;

  // ------------------------------ Start of copied code fragment
  // This code fragment was copied from ApplyFEMCurrentSource.cc 
  // There must be a better way to pass parameters around.
    int numParams=4;

    // Get the electrode parameters input vector and extract the second parameter
    // (element 1) which is the number of electrodes
    // --------------------------------------------------------------------------
    MatrixHandle  hElectrodeParams;

    if (!iportElectrodeParams_->get(hElectrodeParams) || !hElectrodeParams.get_rep()) 
    {
        error("Can't get handle to electrode parameters matrix.");
        return;
    }

    ColumnMatrix* electrodeParams = scinew ColumnMatrix(numParams);
    electrodeParams=dynamic_cast<ColumnMatrix*>(hElectrodeParams.get_rep());

    int L           = (int) ( (*electrodeParams)[1]);
 
    cout << "Number of electrodes is " << L << endl;

  // ------------------------------ End of copied code fragment

  // Allocate space for the output current pattern vector
  ColumnMatrix* currentPattern;
  
  currentPattern = scinew ColumnMatrix(L);

  for (int i=0; i<L; i++) 
  {
    if (k<((L/2)+1))
    {
      (*currentPattern)[i] = currentMagnitude * cos(k*2*PI*i/L);
    }
    else
    {
      int kNew = k-(L/2);
      (*currentPattern)[i] = currentMagnitude * sin(kNew*2*PI*i/L);
    }
  }

  //! Sending result
  oportCurrentVector_->send(MatrixHandle(currentPattern));       


}

void
 TrigCurrentPattern::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace BioPSE


