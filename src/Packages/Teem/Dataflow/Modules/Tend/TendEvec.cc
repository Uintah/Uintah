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
//    File   : TendEvec.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendEvec : public Module {
public:
  TendEvec(SCIRun::GuiContext *ctx);
  virtual ~TendEvec();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiDouble    threshold_;
};

DECLARE_MAKER(TendEvec)

TendEvec::TendEvec(SCIRun::GuiContext *ctx) : 
  Module("TendEvec", ctx, Filter, "Tend", "Teem"), 
  threshold_(ctx->subVar("threshold"))
{
}

TendEvec::~TendEvec() {
}

void 
TendEvec::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  onrrd_ = (NrrdOPort *)get_oport("nout");

  if (!inrrd_) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }
  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;

  int N, sx, sy, sz;
  sx = nin->axis[1].size;
  sy = nin->axis[2].size;
  sz = nin->axis[3].size;
  N = sx*sy*sz;

  NrrdData *nout = new NrrdData();

  nrrdAlloc(nout->nrrd, nrrdTypeFloat, 4, 9, sx, sy, sz);
  if (tenTensorCheck(nin, nrrdTypeFloat, AIR_TRUE)) {
    error("Input Nrrd was not a Tensor field of floats");
    return;
  }

  float *edata = (float *)(nout->nrrd->data);
  float *tdata = (float *)(nin->data);
  float eval[3], evec[9];
  
  float thresh = threshold_.get();
  for (int I=0; I<N; I++) {
    tenEigensolve(eval, evec, tdata);
    float scl = tdata[0] >= thresh;
    for (int cc=0; cc<3; cc++) {
      ELL_3V_SCALE(edata+3*cc, scl, evec+3*cc);
    }
    edata += 9;
    tdata += 7;
  }
  nrrdAxisInfoCopy(nout->nrrd, nin, NULL, NRRD_AXIS_INFO_SIZE_BIT);
  nout->nrrd->axis[0].label = "Unknown:Vector,Unknown:Vector,Unknown:Vector";
  onrrd_->send(NrrdDataHandle(nout));
}

} // End namespace SCITeem
