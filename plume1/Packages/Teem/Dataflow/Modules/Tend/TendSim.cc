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

//    File   : TendMake.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>


namespace SCITeem {

using namespace SCIRun;

class TendSim : public Module {
public:
  TendSim(SCIRun::GuiContext *ctx);
  virtual ~TendSim();
  virtual void execute();

private:
  NrrdIPort*      bmat_;
  NrrdIPort*      referenceimg_;
  NrrdIPort*      tensor_;
  NrrdOPort*      onrrd_;

  GuiDouble       bvalue_;

};

DECLARE_MAKER(TendSim)

TendSim::TendSim(SCIRun::GuiContext *ctx) : 
  Module("TendSim", ctx, Filter, "Tend", "Teem"),
  bvalue_(ctx->subVar("bvalue"))
{
}

TendSim::~TendSim() {
}

void 
TendSim::execute()
{
  NrrdDataHandle bmat_handle;
  NrrdDataHandle referenceimg_handle;
  NrrdDataHandle tensor_handle;
  update_state(NeedData);
  bmat_ = (NrrdIPort *)get_iport("BMatrixNrrd");
  referenceimg_ = (NrrdIPort *)get_iport("ReferenceNrrd");
  tensor_ = (NrrdIPort *)get_iport("TensorNrrd");

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!bmat_->get(bmat_handle))
    return;
  if (!referenceimg_->get(referenceimg_handle))
    return;
  if (!tensor_->get(tensor_handle))
    return;

  if (!bmat_handle.get_rep()) {
    error("Empty input Confidence Nrrd.");
    return;
  }
  if (!referenceimg_handle.get_rep()) {
    error("Empty input ReferenceNrrd Nrrd.");
    return;
  }
  if (!tensor_handle.get_rep()) {
    error("Empty input TensorNrrd Nrrd.");
    return;
  }
  
  Nrrd *bmat = bmat_handle->nrrd;
  Nrrd *referenceimg = referenceimg_handle->nrrd;
  Nrrd *tensor = tensor_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (tenSimulate(nout, referenceimg, tensor, bmat, bvalue_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error simulating tensors: ") + err);
    free(err);
    return;
  }


  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}

} // End namespace SCITeem
