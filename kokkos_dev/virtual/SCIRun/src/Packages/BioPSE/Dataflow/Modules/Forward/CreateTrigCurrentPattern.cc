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

/*
 *  CreateTrigCurrentPattern.cc: Generates a trigonometric current pattern for EIT testing.
 *  
 *  Written by:
 *   lkreda
 *   November 30, 2003
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Math/Trig.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class CreateTrigCurrentPattern : public Module {
public:
  GuiDouble magnitudeTCL_;

  CreateTrigCurrentPattern(GuiContext*);
  virtual ~CreateTrigCurrentPattern();

  virtual void execute();
};

DECLARE_MAKER(CreateTrigCurrentPattern)
  CreateTrigCurrentPattern::CreateTrigCurrentPattern(GuiContext* ctx)
    : Module("CreateTrigCurrentPattern", ctx, Source, "Forward", "BioPSE"),
      magnitudeTCL_(get_ctx()->subVar("magnitudeTCL"))
{
}


CreateTrigCurrentPattern::~CreateTrigCurrentPattern()
{
}


void
CreateTrigCurrentPattern::execute()
{
  double currentMagnitude = Max(magnitudeTCL_.get(), 0.0);

  cout << "Current magnitude = " << currentMagnitude << endl;

  // Get the current pattern index  
  MatrixHandle  hCurrentPatternIndex;
  ColumnMatrix* currPatIdx;
  int           k;

  // -- copy the input current pattern index into local variable, k 

  if (get_input_handle("CurrentPatternIndex", hCurrentPatternIndex, false) &&
      (currPatIdx=dynamic_cast<ColumnMatrix*>(hCurrentPatternIndex.get_rep())) && 
      (currPatIdx->nrows() == 1))
  {
    k = static_cast<int>((*currPatIdx)[0]);
  }
  else{
    msg_stream_ << "The supplied current pattern index is not a 1x1 matrix" << endl;
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
  if (!get_input_handle("Electrode Parameters", hElectrodeParams)) return;

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
      (*currentPattern)[i] = currentMagnitude * cos(k*2*M_PI*i/L);
    }
    else
    {
      int kNew = k-(L/2);
      (*currentPattern)[i] = currentMagnitude * sin(kNew*2*M_PI*i/L);
    }
  }

  //! Sending result
  MatrixHandle cmatrix(currentPattern);
  send_output_handle("CurrentPatternVector", cmatrix);
}

} // End namespace BioPSE


