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
//    File   : TendEval.cc
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

class TendEval : public Module {
public:
  TendEval(SCIRun::GuiContext *ctx);
  virtual ~TendEval();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt       major_;
  GuiInt       medium_;
  GuiInt       minor_;
  GuiDouble    threshold_;
};

DECLARE_MAKER(TendEval)

TendEval::TendEval(SCIRun::GuiContext *ctx) : 
  Module("TendEval", ctx, Filter, "Tend", "Teem"), 
  major_(ctx->subVar("major")),
  medium_(ctx->subVar("medium")),
  minor_(ctx->subVar("minor")),
  threshold_(ctx->subVar("threshold"))
{
}

TendEval::~TendEval() {
}

void 
TendEval::execute()
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

  int compLen=0;
  bool useComp[3];
  if (major_.get()) { useComp[0]=true; compLen++; } else useComp[0]=false;
  if (medium_.get()) { useComp[1]=true; compLen++; } else useComp[1]=false;
  if (minor_.get()) { useComp[2]=true; compLen++; } else useComp[2]=false;

  if (compLen == 0) {
    warning("No eigenvector selected");
    return;
  }

  NrrdData *nout = new NrrdData();

  nrrdAlloc(nout->nrrd, nrrdTypeFloat, 4, compLen, sx, sy, sz);
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
    int seen=0;
    for (int cc=0; cc<3; cc++) {
      if (useComp[cc]) {
	edata[seen] = scl*eval[cc];
	seen++;
      }
    }
    edata += compLen;
    tdata += 7;
  }
  nrrdAxisInfoCopy(nout->nrrd, nin, NULL, NRRD_AXIS_INFO_SIZE_BIT);
  string lname;
  string enames[3] = {"Major", "Medium", "Minor"};
  int init=false;
  for (int i=2; i>=0; i--) {
    if (useComp[i]) { 
      if (init) 
	lname += ",";
      lname += enames[i];
      lname += "Eigenvalue:Scalar";
      init = true;
    }
  }
  strcpy(nout->nrrd->axis[0].label, lname.c_str());
  onrrd_->send(NrrdDataHandle(nout));
}

} // End namespace SCITeem
