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

//    File   : TendEval.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
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
  GuiInt       major_;
  GuiInt       medium_;
  GuiInt       minor_;
  GuiDouble    threshold_;
};

DECLARE_MAKER(TendEval)

TendEval::TendEval(SCIRun::GuiContext *ctx) : 
  Module("TendEval", ctx, Filter, "Tend", "Teem"), 
  major_(get_ctx()->subVar("major"), 1),
  medium_(get_ctx()->subVar("medium"), 0),
  minor_(get_ctx()->subVar("minor"), 0),
  threshold_(get_ctx()->subVar("threshold"), 0.5)
{
}


TendEval::~TendEval()
{
}


void 
TendEval::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("nin", nrrd_handle)) return;

  Nrrd *nin = nrrd_handle->nrrd_;

  int N, sx, sy, sz;
  if (nin->dim > 3) {
    sx = nin->axis[1].size;
    sy = nin->axis[2].size;
    sz = nin->axis[3].size;
    N = sx*sy*sz;
  } else {
    error("Input Nrrd was not 4 dimensions");
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
  size_t size[NRRD_DIM_MAX];
  size[0] = compLen; size[1] = sx;
  size[2] = sy; size[3] = sz;
  nrrdAlloc_nva(nout->nrrd_, nrrdTypeFloat, 4, size);
  if (tenTensorCheck(nin, nrrdTypeFloat, AIR_TRUE, AIR_TRUE)) {
    error("Input Nrrd was not a Tensor field of floats");
    return;
  }

  float *edata = (float *)(nout->nrrd_->data);
  float *tdata = (float *)(nin->data);
  float eval[3], evec[9];
  
  float thresh = threshold_.get();
  for (int I=0; I<N; I++) {
    tenEigensolve_f(eval, evec, tdata);
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
  nrrdAxisInfoCopy(nout->nrrd_, nin, NULL, NRRD_AXIS_INFO_SIZE_BIT);
  NrrdDataHandle ntmp(nout);

  send_output_handle("nout", ntmp);
}

} // End namespace SCITeem
