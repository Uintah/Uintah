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
//    File   : TendAnplot.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

namespace SCITeem {

using namespace SCIRun;

class TendAnplot : public Module {
public:
  TendAnplot(SCIRun::GuiContext *ctx);
  virtual ~TendAnplot();
  virtual void execute();

private:
  NrrdOPort*      onrrd_;

  GuiInt          resolution_;
  GuiInt          whole_;
  GuiInt          values_;
  GuiString       anisotropy_;

  unsigned int get_anisotropy(const string &an);

};

DECLARE_MAKER(TendAnplot)

TendAnplot::TendAnplot(SCIRun::GuiContext *ctx) : 
  Module("TendAnplot", ctx, Filter, "Tend", "Teem"),
  resolution_(ctx->subVar("resolution")),
  whole_(ctx->subVar("whole")),
  values_(ctx->subVar("values")),
  anisotropy_(ctx->subVar("anisotropy"))
{
}

TendAnplot::~TendAnplot() {
}

void 
TendAnplot::execute()
{
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
  Nrrd *nout = nrrdNew();

  if (tenAnisoPlot(nout, get_anisotropy(anisotropy_.get()), 
		   resolution_.get(), whole_.get(), values_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error in TendAnplot: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);

}

unsigned int
TendAnplot::get_anisotropy(const string &an) {
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



