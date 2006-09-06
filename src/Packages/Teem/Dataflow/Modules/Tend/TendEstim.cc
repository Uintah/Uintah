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

//    File   : TendEstim.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendEstim : public Module {
public:
  TendEstim(SCIRun::GuiContext *ctx);
  virtual ~TendEstim();
  virtual void execute();

private:
  NrrdIPort*      inbmat_;
  NrrdIPort*      indwi_;

  GuiInt       knownB0_;
  GuiInt       use_default_threshold_;
  GuiDouble    threshold_;
  GuiDouble    soft_;
  GuiDouble    scale_;
};

DECLARE_MAKER(TendEstim)

TendEstim::TendEstim(SCIRun::GuiContext *ctx) : 
  Module("TendEstim", ctx, Filter, "Tend", "Teem"), 
  knownB0_(get_ctx()->subVar("knownB0"), 1),
  use_default_threshold_(get_ctx()->subVar("use-default-threshold"), 1),
  threshold_(get_ctx()->subVar("threshold"), 0.0),
  soft_(get_ctx()->subVar("soft"), 0.0),
  scale_(get_ctx()->subVar("scale"), 1.0)
{
}


TendEstim::~TendEstim()
{
}


void 
TendEstim::execute()
{
  NrrdDataHandle bmat_handle;
  NrrdDataHandle dwi_handle;
  update_state(NeedData);
  inbmat_ = (NrrdIPort *)get_iport("Bmat");
  indwi_ = (NrrdIPort *)get_iport("DWI");
  //  oerr_ = (NrrdOPort *)get_oport("Error");

  //Nrrd *sliced_bmat = 0;
  if (inbmat_->get(bmat_handle)){
    if (!bmat_handle.get_rep()) {
      error("Empty input Bmat Nrrd.");
      return;
    }
  }
  if (!indwi_->get(dwi_handle))
    return;

  if (!bmat_handle.get_rep()) {
    error("Empty input Bmat Nrrd.");
    return;
  }
  if (!dwi_handle.get_rep()) {
    error("Empty input DWI Nrrd.");
    return;
  }

 
  Nrrd *nout = nrrdNew();
  float threshold;
  if (use_default_threshold_.get()) threshold = AIR_NAN;
  else threshold = threshold_.get();

  int knownB0 = knownB0_.get(); // TRUE for brains, FALSE for dog hearts
  Nrrd* dummy = nrrdNew();
  if (tenEstimateLinear4D(nout, NULL, &dummy, dwi_handle->nrrd_, 
			  bmat_handle->nrrd_, knownB0, threshold, 
			  soft_.get(), scale_.get()))
  {
    char *err = biffGetDone(TEN);
    error(string("Error in TendEstim: ") + err);
    free(err);
    return;
  }
  nrrdNuke(dummy);

  nout->axis[0].kind = nrrdKind3DMaskedSymMatrix;
  NrrdDataHandle ntmp(scinew NrrdData(nout));

  send_output_handle("Tensors", ntmp);

  update_state(Completed);
}

} // End namespace SCITeem
