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
//    File   : TendEvq.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

namespace SCITeem {

using namespace SCIRun;

class TendEvq : public Module {
public:
  TendEvq(SCIRun::GuiContext *ctx);
  virtual ~TendEvq();
  virtual void execute();

private:
  NrrdIPort *     inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          index_;
  GuiString       anisotropy_;
  GuiInt          ns_;

  unsigned int get_anisotropy(const string &an);

};

DECLARE_MAKER(TendEvq)

TendEvq::TendEvq(SCIRun::GuiContext *ctx) : 
  Module("TendEvq", ctx, Filter, "Tend", "Teem"),
  index_(ctx->subVar("index")),
  anisotropy_(ctx->subVar("anisotropy")),
  ns_(ctx->subVar("ns"))
{
}

TendEvq::~TendEvq() {
}

void 
TendEvq::execute()
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

  if (tenEvqVolume(nout, nin, index_.get(), get_anisotropy(anisotropy_.get()), ns_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error quantizing directions of diffusions: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);

}

unsigned int
TendEvq::get_anisotropy(const string &an) {
  if (an == "cl1")
    return tenAniso_Cl1;
  else if (an == "cl2")
    return tenAniso_Cl2;
  else if (an == "cp1")
    return tenAniso_Cp1;
  else if (an == "cp2")
    return tenAniso_Cp2;
  else if (an == "ca1")
    return tenAniso_Ca1;
  else if (an == "ca2")
    return tenAniso_Ca2;
  else if (an == "cs1")
    return tenAniso_Cs1;
  else if (an == "cs2")
    return tenAniso_Cs2;
  else if (an == "ct1")
    return tenAniso_Ct1;
  else if (an == "ct2")
    return tenAniso_Ct1;
  else if (an == "ra")
    return tenAniso_RA;
  else if (an == "fa")
    return tenAniso_FA;
  else if (an == "vf")
    return tenAniso_VF;
  else if (an == "tr")
    return tenAniso_Tr;
  else {
    error("Unkown anisotropy metric.  Using trace");
    return tenAniso_Tr;
  }
  
}

} // End namespace SCITeem


