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

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendMake : public Module {
public:
  TendMake(SCIRun::GuiContext *ctx);
  virtual ~TendMake();
  virtual void execute();

private:
  NrrdIPort*      indwi_;
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
  NrrdDataHandle dwi_handle;
//   NrrdDataHandle eval_handle;
//   NrrdDataHandle evec_handle;
  update_state(NeedData);
  indwi_ = (NrrdIPort *)get_iport("DWI");
  //  inevals_ = (NrrdIPort *)get_iport("Evals");
  //inevecs_ = (NrrdIPort *)get_iport("Evecs");

  onrrd_ = (NrrdOPort *)get_oport("nout");

  if (!indwi_) {
    error("Unable to initialize iport 'DWI'.");
    return;
  }
//   if (!inevals_) {
//     error("Unable to initialize iport 'Evals'.");
//     return;
//   }
//   if (!inevecs_) {
//     error("Unable to initialize iport 'Evecs'.");
//     return;
//   }
  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }
  if (!indwi_->get(dwi_handle))
    return;
//   if (!inevals_->get(eval_handle))
//     return;
//   if (!inevecs_->get(evec_handle))
//     return;

  if (!dwi_handle.get_rep()) {
    error("Empty input DWI Nrrd.");
    return;
  }
//   if (!eval_handle.get_rep()) {
//     error("Empty input Evals Nrrd.");
//     return;
//   }
//   if (!evec_handle.get_rep()) {
//     error("Empty input Evecs Nrrd.");
//     return;
//   }

  //  Nrrd *nin = nrrd_handle->nrrd;

  error("This module is a stub.  Implement me.");

  //onrrd_->send(NrrdDataHandle(nrrd_joined));
}

} // End namespace SCITeem
