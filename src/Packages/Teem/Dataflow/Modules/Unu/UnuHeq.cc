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
//    File   : UnuHeq.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class UnuHeq : public Module {
public:
  UnuHeq(SCIRun::GuiContext *ctx);
  virtual ~UnuHeq();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt       bins_;
  GuiInt       sbins_;
  GuiDouble    amount_;
  
};

DECLARE_MAKER(UnuHeq)

UnuHeq::UnuHeq(SCIRun::GuiContext *ctx) : 
  Module("UnuHeq", ctx, Filter, "Unu", "Teem"),
  bins_(ctx->subVar("bins")),
  sbins_(ctx->subVar("sbins")),
  amount_(ctx->subVar("amount"))
{
}

UnuHeq::~UnuHeq() {
}

void 
UnuHeq::execute()
{
  NrrdDataHandle nrrd_handle;

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


  if (!nrrd_handle.get_rep()) 
    return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (nrrdHistoEq(nout, nin, NULL, bins_.get(), sbins_.get(), amount_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error creating Heqing nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}

} // End namespace SCITeem


