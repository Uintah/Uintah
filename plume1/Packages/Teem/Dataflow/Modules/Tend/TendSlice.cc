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

//    File   : TendSlice.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

namespace SCITeem {

using namespace SCIRun;

class TendSlice : public Module {
public:
  TendSlice(SCIRun::GuiContext *ctx);
  virtual ~TendSlice();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          axis_;
  GuiInt          position_;
  GuiInt          dimension_;

};

DECLARE_MAKER(TendSlice)

TendSlice::TendSlice(SCIRun::GuiContext *ctx) : 
  Module("TendSlice", ctx, Filter, "Tend", "Teem"),
  axis_(ctx->subVar("axis")),
  position_(ctx->subVar("position")),
  dimension_(ctx->subVar("dimension"))
{
}

TendSlice::~TendSlice() {
}

void 
TendSlice::execute()
{
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input InputNrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (tenSlice(nout, nin, axis_.get(), position_.get(), dimension_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error slicing 3D tensor: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}

} // End namespace SCITeem
