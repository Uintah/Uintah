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

class TendMake : public Module {
public:
  TendMake(SCIRun::GuiContext *ctx);
  virtual ~TendMake();
  virtual void execute();

private:
  NrrdIPort*      inconfidence_;
  NrrdIPort*      inevals_;
  NrrdIPort*      inevecs_;
  NrrdOPort*      onrrd_;

};

DECLARE_MAKER(TendMake)

TendMake::TendMake(SCIRun::GuiContext *ctx) : 
  Module("TendMake", ctx, Filter, "Tend", "Teem")
{
}

TendMake::~TendMake() {
}

void 
TendMake::execute()
{
  NrrdDataHandle conf_handle;
  NrrdDataHandle eval_handle;
  NrrdDataHandle evec_handle;
  update_state(NeedData);
  inconfidence_ = (NrrdIPort *)get_iport("Confidence");
  inevals_ = (NrrdIPort *)get_iport("Evals");
  inevecs_ = (NrrdIPort *)get_iport("Evecs");

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inconfidence_) {
    error("Unable to initialize iport 'Confidence'.");
    return;
  }
  if (!inevals_) {
    error("Unable to initialize iport 'Evals'.");
    return;
  }
  if (!inevecs_) {
    error("Unable to initialize iport 'Evecs'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
  if (!inconfidence_->get(conf_handle))
    return;
  if (!inevals_->get(eval_handle))
    return;
  if (!inevecs_->get(evec_handle))
    return;

  if (!conf_handle.get_rep()) {
    error("Empty input Confidence Nrrd.");
    return;
  }
  if (!eval_handle.get_rep()) {
    error("Empty input Evals Nrrd.");
    return;
  }
  if (!evec_handle.get_rep()) {
    error("Empty input Evecs Nrrd.");
    return;
  }

  Nrrd *confidence = conf_handle->nrrd;
  Nrrd *eval = eval_handle->nrrd;
  Nrrd *evec = evec_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (tenMake(nout, confidence, eval, evec)) {
    char *err = biffGetDone(TEN);
    error(string("Error creating DT volume: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);

}

} // End namespace SCITeem
