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

//    File   : TendEvalClamp.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendEvalClamp : public Module {
public:
  TendEvalClamp(SCIRun::GuiContext *ctx);
  virtual ~TendEvalClamp();
  virtual void execute();

private:
  GuiString       min_;
  GuiString       max_;
};

DECLARE_MAKER(TendEvalClamp)

TendEvalClamp::TendEvalClamp(SCIRun::GuiContext *ctx) : 
  Module("TendEvalClamp", ctx, Filter, "Tend", "Teem"), 
  min_(get_ctx()->subVar("min"), "0.0001"),
  max_(get_ctx()->subVar("max"), "NaN")
{
}

TendEvalClamp::~TendEvalClamp() {
}
void 
TendEvalClamp::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("nin", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();
  
  float min, max;
  min=max=AIR_NAN;

  if (min_.get() != "NaN" && min_.get() != "nan") 
    min = atof(min_.get().c_str());
  if (max_.get() != "NaN" && max_.get() != "nan")
    max = atof(max_.get().c_str());

  if (tenEigenvalueClamp(nout, nin, min, max)) {
    char *err = biffGetDone(TEN);
    error(string("Error making tendEvalClamp volume: ") + err);
    free(err);
    return;
  }

  NrrdDataHandle ntmp(scinew NrrdData(nout));

  send_output_handle("nout", ntmp);
}

} // End namespace SCITeem
