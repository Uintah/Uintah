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

//    File   : TendEpireg.cc
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

class TendEpireg : public Module {
public:
  TendEpireg(SCIRun::GuiContext *ctx);
  virtual ~TendEpireg();
  virtual void execute();
  virtual void presave();

private:
  bool extract_gradients(vector<double> &d);

  NrrdIPort*      inrrd_;
  NrrdIPort*      igrad_; 
  NrrdOPort*      onrrd_;

  GuiString    gradient_list_;
  GuiInt       reference_;
  GuiDouble    blur_x_;
  GuiDouble    blur_y_;
  GuiInt       use_default_threshold_;
  GuiDouble    threshold_;
  GuiInt       cc_analysis_;
  GuiDouble    fitting_;
  GuiString    kernel_;
  GuiDouble    sigma_;
  GuiDouble    extent_;
};

DECLARE_MAKER(TendEpireg)

TendEpireg::TendEpireg(SCIRun::GuiContext *ctx) : 
  Module("TendEpireg", ctx, Filter, "Tend", "Teem"), 
  gradient_list_(ctx->subVar("gradient_list")),
  reference_(ctx->subVar("reference")),
  blur_x_(ctx->subVar("blur_x")),
  blur_y_(ctx->subVar("blur_y")),
  use_default_threshold_(ctx->subVar("use-default-threshold")),
  threshold_(ctx->subVar("threshold")),
  cc_analysis_(ctx->subVar("cc_analysis")),
  fitting_(ctx->subVar("fitting")),
  kernel_(ctx->subVar("kernel")),
  sigma_(ctx->subVar("sigma")),
  extent_(ctx->subVar("extent"))
{
}

TendEpireg::~TendEpireg() {
}



// Create a memory for a new nrrd, that is arranged 3 x n;
bool
TendEpireg::extract_gradients(vector<double> &d)
{
  gui->execute(id + " update_text"); // make gradient_list current
  istringstream str(gradient_list_.get().c_str());
  for (;;)
  {
    double tmp;
    str >> tmp;
    if (!str.eof() && !str.fail()) {
      d.push_back(tmp);
    }
    else {
      break;
    }
  }
  if (d.size() % 3 != 0) {
    error("Error: Number of input values must be divisible by 3");
    return false;
  }
  return true;
}


void 
TendEpireg::execute()
{
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle grad_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  igrad_ = (NrrdIPort *)get_iport("ngrad");
  onrrd_ = (NrrdOPort *)get_oport("nout");

  bool we_own_the_data;
  vector<double> *mat=0;

  if (!inrrd_->get(nrrd_handle))
    return;
  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }
  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *ngrad;

  if (igrad_->get(grad_handle) && grad_handle.get_rep()) {
    we_own_the_data = false;
    ngrad = grad_handle->nrrd;
  } else {
    we_own_the_data = false;
    mat = new vector<double>;
    if (! extract_gradients(*mat)) {
      error("Please adjust your input in the gui to represent a 3 x N set.");
      return;
    }
    ngrad = nrrdNew();
    nrrdWrap(ngrad, &(*mat)[0], nrrdTypeDouble, 2, 3, (*mat).size() / 3);
  }

  reset_vars();

  NrrdKernel *kern;
  double p[NRRD_KERNEL_PARMS_NUM];
  memset(p, 0, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  p[0] = 1.0;
  if (kernel_.get() == "box") {
    kern = nrrdKernelBox;
  } else if (kernel_.get() == "tent") {
    kern = nrrdKernelTent;
  } else if (kernel_.get() == "gaussian") { 
    kern = nrrdKernelGaussian; 
    p[1] = sigma_.get(); 
    p[2] = extent_.get(); 
  } else if (kernel_.get() == "cubicCR") { 
    kern = nrrdKernelBCCubic; 
    p[1] = 0; 
    p[2] = 0.5; 
  } else if (kernel_.get() == "cubicBS") { 
    kern = nrrdKernelBCCubic; 
    p[1] = 1; 
    p[2] = 0; 
  } else if (kernel_.get() == "hann") { 
    kern = nrrdKernelHann;
    p[1] = 8; 
  } else  { // default is quartic
    kern = nrrdKernelAQuartic; 
    p[1] = 0.0834; 
  }

  Nrrd *nout = nrrdNew();
  float threshold;
  if (use_default_threshold_.get()) threshold = AIR_NAN;
  else threshold = threshold_.get();
//  cerr << "threshold = "<<threshold<<"\n";
  if (tenEpiRegister4D(nout, nin, ngrad, reference_.get(),
		       blur_x_.get(), blur_y_.get(), fitting_.get(), 
		       threshold, cc_analysis_.get(),
		       kern, p, 0, 0)) {
    char *err = biffGetDone(TEN);
    error(string("Error in epireg: ") + err);
    free(err);
    return;
  }
      
  if (we_own_the_data) {
    nrrdNix(ngrad);
    delete mat;
  }

  NrrdData *output = scinew NrrdData;
  output->nrrd = nout;
  //output->copy_sci_data(*nrrd_handle.get_rep());
  //output->nrrd->axis[0].label = airStrdup(nin->axis[0].label);
  onrrd_->send(NrrdDataHandle(output));

  update_state(Completed);
}


void
TendEpireg::presave()
{
  gui->execute(id + " update_text"); // make gradient_list current
}


} // End namespace SCITeem
