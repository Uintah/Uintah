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
//    File   : TendSatin.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

extern "C" {
  int tend_satinGen(Nrrd *nout, float parm, float mina, float maxa, int wsize,
		    float thick, float bnd, int torus);
}

namespace SCITeem {

using namespace SCIRun;

class TendSatin : public Module {
public:
  TendSatin(SCIRun::GuiContext *ctx);
  virtual ~TendSatin();
  virtual void execute();

private:
  NrrdOPort*      onrrd_;

  GuiInt          torus_;
  GuiDouble       anisotropy_;
  GuiDouble       maxca1_;
  GuiDouble       minca1_;
  GuiDouble       boundary_;
  GuiDouble       thickness_;
  GuiInt          size_;

};

DECLARE_MAKER(TendSatin)

TendSatin::TendSatin(SCIRun::GuiContext *ctx) : 
  Module("TendSatin", ctx, Filter, "Tend", "Teem"),
  torus_(ctx->subVar("torus")),
  anisotropy_(ctx->subVar("anisotropy")),
  maxca1_(ctx->subVar("maxca1")),
  minca1_(ctx->subVar("minca1")),
  boundary_(ctx->subVar("boundary")),
  thickness_(ctx->subVar("thickness")),
  size_(ctx->subVar("size"))
{
}

TendSatin::~TendSatin() {
}

void 
TendSatin::execute()
{
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
  Nrrd *nout = nrrdNew();

  if (tend_satinGen(nout, anisotropy_.get(), minca1_.get(), 
		    maxca1_.get(), size_.get(), thickness_.get(), 
		    boundary_.get(), torus_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error in TendSatin: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  nrrd->nrrd->axis[0].kind = nrrdKind3DMaskedSymTensor;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);


}

} // End namespace SCITeem



