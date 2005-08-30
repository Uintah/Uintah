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

//    File   : UnuGamma.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class UnuGamma : public Module {
public:
  UnuGamma(SCIRun::GuiContext *ctx);
  virtual ~UnuGamma();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiDouble    gamma_;
  GuiDouble    min_;
  GuiInt       useinputmin_;
  GuiDouble    max_;
  GuiInt       useinputmax_;
};

DECLARE_MAKER(UnuGamma)

UnuGamma::UnuGamma(SCIRun::GuiContext *ctx) : 
  Module("UnuGamma", ctx, Filter, "UnuAtoM", "Teem"), 
  gamma_(ctx->subVar("gamma")),
  min_(ctx->subVar("min")),
  useinputmin_(ctx->subVar("useinputmin")),
  max_(ctx->subVar("max")),
  useinputmax_(ctx->subVar("useinputmax"))
{
}

UnuGamma::~UnuGamma() {
}

void 
UnuGamma::execute()
{
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle weight_handle;

  update_state(NeedData);

  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle)) 
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  NrrdRange *range = NULL;

  double min = AIR_NAN, max = AIR_NAN;
  if (!useinputmin_.get())
    min = min_.get();
  if (!useinputmax_.get())
    max = max_.get();

  nrrdRangeNew(min, max);
  nrrdRangeSafeSet(range, nin, nrrdBlind8BitRangeState);

  if (nrrdArithGamma(nout, nin, range, gamma_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error creating peforming unu gamma on nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  nrrdKeyValueCopy(nout, nin);
  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}


} // End namespace SCITeem

