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
//    File   : UnuGamma.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

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
  GuiDouble    max_;
};

DECLARE_MAKER(UnuGamma)

UnuGamma::UnuGamma(SCIRun::GuiContext *ctx) : 
  Module("UnuGamma", ctx, Filter, "Unu", "Teem"), 
  gamma_(ctx->subVar("gamma")),
  min_(ctx->subVar("min")),
  max_(ctx->subVar("max"))
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

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
  if (!inrrd_->get(nrrd_handle)) 
    return;


  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  NrrdRange *range = nrrdRangeNew(min_.get(), max_.get());
  nrrdRangeSafeSet(range, nin, nrrdBlind8BitRangeState);

  if (nrrdArithGamma(nout, nin, range, gamma_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error creating peforming unu gamma on nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}


} // End namespace SCITeem

