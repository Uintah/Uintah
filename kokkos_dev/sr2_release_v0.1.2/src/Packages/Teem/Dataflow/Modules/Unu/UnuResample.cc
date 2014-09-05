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


/*
 *  UnuResample
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <sci_defs/bits_defs.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

#include <iostream>
#include <stdio.h>

using std::endl;

namespace SCITeem {
using namespace SCIRun;

class UnuResample : public Module {
public:
  int getint(const char *str, size_t *n, int *none);
  UnuResample(GuiContext *ctx);
  virtual ~UnuResample();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString           filtertype_;
  GuiInt              dim_;
  GuiInt              uis_;
  vector<GuiString*>  resampAxes_;
  vector<string>      last_RA_;
  GuiDouble           sigma_;
  GuiDouble           extent_;
  string              last_filtertype_;
  int                 last_generation_;
  NrrdDataHandle      last_nrrdH_;

};

} // End namespace SCITeem
using namespace SCITeem;

DECLARE_MAKER(UnuResample)

UnuResample::UnuResample(GuiContext *ctx) : 
  Module("UnuResample", ctx, Filter, "UnuNtoZ", "Teem"),
  filtertype_(get_ctx()->subVar("filtertype")),
  dim_(get_ctx()->subVar("dim")),
  uis_(get_ctx()->subVar("uis")),
  sigma_(get_ctx()->subVar("sigma")),
  extent_(get_ctx()->subVar("extent")),
  last_filtertype_(""), 
  last_generation_(-1), 
  last_nrrdH_(0)
{
  // value will be overwritten at gui side initialization.
  dim_.set(0);

  last_RA_.resize(4, "--");
  for (int a = 0; a < 4; a++) {
    ostringstream str;
    str << "resampAxis" << a;
    resampAxes_.push_back(new GuiString(get_ctx()->subVar(str.str())));
  }
}

UnuResample::~UnuResample() 
{
}


int 
UnuResample::getint(const char *str, size_t *n, int *none) 
{
  if (!strlen(str)) return 1;
  if (str[0] == 'x') {
    if (strlen(str) > 1) {
      double ratio;
      if (sscanf(str+1, "%lf", &ratio) != 1) return 1;
      *n = (int)(*n * ratio);
    }
  } else if (str[0] == '=') {
    *none = 1;
  } else {
#ifndef SCI_64BITS
    if (sscanf(str, "%d", n) != 1) return 1;
#else
    if (sscanf(str, "%ld", n) != 1) return 1;
#endif
  }
  if (*n < 2 && !none) {
    error("Invalid # of samples (" + to_string(*n) + ").");
    return 1;
  }
  return 0;
}

void 
UnuResample::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrdH;
  if (!get_input_handle("Nrrd", nrrdH)) return;

  reset_vars();

  bool new_dataset = (last_generation_ != nrrdH->generation);
  bool first_time = (last_generation_ == -1);

  // Create any resample axes that might have been saved.
  if (first_time)
  {
    uis_.reset();
    for (int i=4; i<uis_.get(); i++)
    {
      ostringstream str, str2;
      str << "resampAxis" << i;
      str2 << i;
      resampAxes_.push_back(new GuiString(get_ctx()->subVar(str.str())));
      last_RA_.push_back(resampAxes_[i]->get());
      get_gui()->execute(get_id() +" make_min_max " + str2.str());
    }
  }

  last_generation_ = nrrdH->generation;
  dim_.set(nrrdH->nrrd_->dim);
  dim_.reset();

  // remove any unused uis or add any needes uis
  if ((unsigned int) uis_.get() > nrrdH->nrrd_->dim) {
    // remove them
    for(unsigned int i=uis_.get()-1; i>=nrrdH->nrrd_->dim; i--) {
      ostringstream str;
      str << i;
      vector<GuiString*>::iterator iter = resampAxes_.end();
      vector<string>::iterator iter2 = last_RA_.end();
      resampAxes_.erase(iter, iter);
      last_RA_.erase(iter2, iter2);
      get_gui()->execute(get_id().c_str() + string(" clear_axis " + str.str()));
    }
    uis_.set(nrrdH->nrrd_->dim);
  } else if ((unsigned int) uis_.get() < nrrdH->nrrd_->dim) {
    for (int i=uis_.get()-1; i< dim_.get(); i++) {
      ostringstream str, str2;
      str << "resampAxis" << i;
      str2 << i;
      resampAxes_.push_back(new GuiString(get_ctx()->subVar(str.str())));
      last_RA_.push_back("x1");
      get_gui()->execute(get_id().c_str() + string(" make_min_max " + str2.str()));
    }
    uis_.set(nrrdH->nrrd_->dim);
  }

  filtertype_.reset();
  for (int a = 0; a < dim_.get(); a++) {
    resampAxes_[a]->reset();
  }

  // See if gui values have changed from last execute,
  // and set up execution values. 
  bool changed = false;
  if (last_filtertype_ != filtertype_.get()) {
    changed = true;
    last_filtertype_ = filtertype_.get();
  }
  for (int a = 0; a < dim_.get(); a++) {
    if (last_RA_[a] != resampAxes_[a]->get()) {
      changed = true;
      last_RA_[a] = resampAxes_[a]->get();
    }
  }

  if (!changed && !new_dataset && last_nrrdH_.get_rep()) {
    send_output_handle("Nrrd", last_nrrdH_, true);
    return;
  }
  
  NrrdResampleInfo *info = nrrdResampleInfoNew();

  Nrrd *nin = nrrdH->nrrd_;
  remark("Resampling with a " + last_filtertype_ + " filter.");
  NrrdKernel *kern;
  double p[NRRD_KERNEL_PARMS_NUM];
  memset(p, 0, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  p[0] = 1.0;
  if (last_filtertype_ == "box") {
    kern = nrrdKernelBox;
  } else if (last_filtertype_ == "tent") {
    kern = nrrdKernelTent;
  } else if (last_filtertype_ == "cubicCR") { 
    kern = nrrdKernelBCCubic; 
    p[1] = 0; 
    p[2] = 0.5; 
  } else if (last_filtertype_ == "cubicBS") { 
    kern = nrrdKernelBCCubic; 
    p[1] = 1; 
    p[2] = 0; 
  } else if (last_filtertype_ == "gaussian") { 
    kern = nrrdKernelGaussian; 
    p[0] = sigma_.get(); 
    p[1] = extent_.get(); 
  } else  { // default is quartic
    kern = nrrdKernelAQuartic; 
    p[1] = 0.0834; // most accurate as per Teem documenation
  }

  for (int a = 0; a < dim_.get(); a++) {
    info->kernel[a] = kern;
    msg_stream_ << "NrrdResample sizes: ";
    info->samples[a]=nin->axis[a].size;
    char *str = airStrdup(resampAxes_[a]->get().c_str());
    if (nrrdKindSize(nin->axis[a].kind) > 1 && str != "=") {
      warning("Trying to resample along axis " + to_string(a) + " which is not of nrrdKindDomain or nrrdKindUnknown.");
    }
    int none=0;
    if (getint(str, &(info->samples[a]), &none)) {
      error("NrrdResample -- bad size."); 
      return;
    }
    if (none) info->kernel[a] = 0;
    msg_stream_ << info->samples[a];
    if (!info->kernel[a]) msg_stream_ << "=";
    msg_stream_ << " ";

    memcpy(info->parm[a], p, NRRD_KERNEL_PARMS_NUM * sizeof(double));
    if (info->kernel[a] && 
	(!(airExists(nin->axis[a].min) && airExists(nin->axis[a].max)))) {
      nrrdAxisInfoMinMaxSet(nrrdH->nrrd_, a, nin->axis[a].center ? 
			nin->axis[a].center : nrrdDefaultCenter);
    }
    info->min[a] = nrrdH->nrrd_->axis[a].min;
    info->max[a] = nrrdH->nrrd_->axis[a].max;
  }    
  msg_stream_ << endl;
  info->boundary = nrrdBoundaryBleed;
  info->type = nin->type;
  info->renormalize = AIR_TRUE;

  last_generation_ = nrrdH->generation;

  NrrdData *nrrd = scinew NrrdData;
  if (nrrdSpatialResample(nrrd->nrrd_, nin, info)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") +  err);
    msg_stream_ << "  input Nrrd: nin->dim=" << nin->dim << "\n";
    free(err);
  }
  nrrdResampleInfoNix(info); 
  //nrrd->copy_sci_data(*nrrdH.get_rep());
  last_nrrdH_ = nrrd;

  send_output_handle("Nrrd", last_nrrdH_, true);
}


void 
UnuResample::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("UnuResample needs a minor command");
    return;
  }

  if( args[1] == "add_axis" ) 
  {
      uis_.reset();
      int i = uis_.get();
      ostringstream str, str2;
      str << "resampAxis" << i;
      str2 << i;
      resampAxes_.push_back(new GuiString(get_ctx()->subVar(str.str())));
      last_RA_.push_back("x1");
      get_gui()->execute(get_id().c_str() + string(" make_min_max " + str2.str()));
      uis_.set(uis_.get() + 1);
  }
  else if( args[1] == "remove_axis" ) 
  {
    uis_.reset();
    int i = uis_.get()-1;
    ostringstream str;
    str << i;
    vector<GuiString*>::iterator iter = resampAxes_.end();
    vector<string>::iterator iter2 = last_RA_.end();
    resampAxes_.erase(iter, iter);
    last_RA_.erase(iter2, iter2);
    get_gui()->execute(get_id().c_str() + string(" clear_axis " + str.str()));
    uis_.set(uis_.get() - 1);
  }
  else 
  {
    Module::tcl_command(args, userdata);
  }
}

