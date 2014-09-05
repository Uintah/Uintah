//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : TendMake.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
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

  if (!bmat_) {
    error("Unable to initialize iport 'BMatrixNrrd'.");
    return;
  }
  if (!referenceimg_) {
    error("Unable to initialize iport 'ReferenceNrrd'.");
    return;
  }
  if (!tensor_) {
    error("Unable to initialize iport 'TensorNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
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
