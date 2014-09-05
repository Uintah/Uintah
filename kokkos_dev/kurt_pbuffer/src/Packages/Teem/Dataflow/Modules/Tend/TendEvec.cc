/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

//    File   : TendEvec.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
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

  GuiInt       major_;
  GuiInt       medium_;
  GuiInt       minor_;
  GuiDouble    threshold_;
};

DECLARE_MAKER(TendEvec)

TendEvec::TendEvec(SCIRun::GuiContext *ctx) : 
  Module("TendEvec", ctx, Filter, "Tend", "Teem"), 
  major_(ctx->subVar("major")),
  medium_(ctx->subVar("medium")),
  minor_(ctx->subVar("minor")),
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

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;

  int N, sx, sy, sz;
  if (nin->dim > 3) {
    sx = nin->axis[1].size;
    sy = nin->axis[2].size;
    sz = nin->axis[3].size;
    N = sx*sy*sz;
  } else {
    error("Input Nrrd was not 4D");
    return;
  }

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

  nrrdAlloc(nout->nrrd, nrrdTypeFloat, 4, 3*compLen, sx, sy, sz);
  if (tenTensorCheck(nin, nrrdTypeFloat, AIR_TRUE, AIR_TRUE)) {
    error("Input Nrrd was not a Tensor field of floats");
    return;
  }

  float *edata = (float *)(nout->nrrd->data);
  float *tdata = (float *)(nin->data);
  float eval[3], evec[9];
  
  float thresh = threshold_.get();
  for (int I=0; I<N; I++) {
    tenEigensolve_f(eval, evec, tdata);
    float scl = tdata[0] >= thresh;
    int seen=0;
    for (int cc=0; cc<3; cc++) {
      if (useComp[cc]) {
	ELL_3V_SCALE(edata+3*seen, scl, evec+3*cc);
	seen++;
      }
    }
    edata += 3*compLen;
    tdata += 7;
  }
  nrrdAxisInfoCopy(nout->nrrd, nin, NULL, NRRD_AXIS_INFO_SIZE_BIT);
//   string lname;
//   string enames[3] = {"Major", "Medium", "Minor"};
//   int init=false;
//   for (int i=2; i>=0; i--) {
//     if (useComp[i]) { 
//       if (init) 
// 	lname += ",";
//       lname += enames[i];
//       lname += "Eigenvector:Vector";
//       init = true;
//     }
//   }

//   strcpy(nout->nrrd->axis[0].label, lname.c_str());
  onrrd_->send(NrrdDataHandle(nout));
}

} // End namespace SCITeem
