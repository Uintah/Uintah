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

//    File   : TendNorm.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendNorm : public Module {
public:
  TendNorm(SCIRun::GuiContext *ctx);
  virtual ~TendNorm();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiDouble       major_weight_;
  GuiDouble       medium_weight_;
  GuiDouble       minor_weight_;
  GuiDouble       amount_;
  GuiDouble       target_;
};

DECLARE_MAKER(TendNorm)

TendNorm::TendNorm(SCIRun::GuiContext *ctx) : 
  Module("TendNorm", ctx, Filter, "Tend", "Teem"), 
  major_weight_(ctx->subVar("major-weight")),
  medium_weight_(ctx->subVar("medium-weight")),
  minor_weight_(ctx->subVar("minor-weight")),
  amount_(ctx->subVar("amount")),
  target_(ctx->subVar("target"))
{
}

TendNorm::~TendNorm() {
}
void 
TendNorm::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  onrrd_ = (NrrdOPort *)get_oport("nout");

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }
  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  double weights[3];
  weights[0]=major_weight_.get();
  weights[1]=medium_weight_.get();
  weights[2]=minor_weight_.get();

  if (tenSizeNormalize(nout, nin, weights, (float)amount_.get(), target_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error in TendNorm: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  //nrrd->copy_sci_data(*nrrd_handle.get_rep());
  onrrd_->send(NrrdDataHandle(nrrd));
}

} // End namespace SCITeem
