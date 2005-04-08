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

//    File   : TendSatin.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
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

  nrrd->nrrd->axis[0].kind = nrrdKind3DMaskedSymMatrix;

  NrrdDataHandle out(nrrd);

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");
  onrrd_->send(out);
}


} // End namespace SCITeem



