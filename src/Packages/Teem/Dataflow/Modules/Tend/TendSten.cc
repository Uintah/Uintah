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
//    File   : TendSten.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

namespace SCITeem {

using namespace SCIRun;

class TendSten : public Module {
public:
  TendSten(SCIRun::GuiContext *ctx);
  virtual ~TendSten();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          diffscale_;
  GuiInt          intscale_;
  GuiInt          factor_;

};

DECLARE_MAKER(TendSten)

TendSten::TendSten(SCIRun::GuiContext *ctx) : 
  Module("TendSten", ctx, Filter, "Tend", "Teem"),
  diffscale_(ctx->subVar("diffscale")),
  intscale_(ctx->subVar("intscale")),
  factor_(ctx->subVar("factor"))
{
}

TendSten::~TendSten() {
}

void 
TendSten::execute()
{
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_) {
    error("Unable to initialize iport  'InputNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input InputNrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (gageStructureTensor(nout, nin, diffscale_.get(), intscale_.get(), 
			  factor_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error calculating tensors: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);

}

} // End namespace SCITeem

