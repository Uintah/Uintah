/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  NrrdResample
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>

namespace SCITeem {

class NrrdResample : public Module {
  NrrdIPort* inrrd_;
  NrrdOPort* onrrd_;
  GuiString filtertype_;
  GuiString resampAxis0_;
  GuiString resampAxis1_;
  GuiString resampAxis2_;
  string last_filtertype_;
  string last_resampAxis0_;
  string last_resampAxis1_;
  string last_resampAxis2_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  int getint(const char *str, int *n, int *none);
  int get_sizes(string *resampAxis, Nrrd *nrrd, nrrdResampleInfo *info);
  NrrdResample(const string& id);
  virtual ~NrrdResample();
  virtual void execute();
};

extern "C" Module* make_NrrdResample(const string& id)
{
    return new NrrdResample(id);
}

NrrdResample::NrrdResample(const string& id)
  : Module("NrrdResample", id, Filter, "Filters", "Teem"), filtertype_("filtertype", id, this),
    resampAxis0_("resampAxis0", id, this), 
    resampAxis1_("resampAxis1", id, this), 
    resampAxis2_("resampAxis2", id, this), 
    last_filtertype_(""), last_resampAxis0_(""), last_resampAxis1_(""),
    last_resampAxis2_(""), last_generation_(-1), last_nrrdH_(0)
{
}

NrrdResample::~NrrdResample() {
}

int NrrdResample::getint(const char *str, int *n, int *none) {
  if (!strlen(str)) return 1;
  if (str[0] == 'x') {
    if (strlen(str) > 1) {
      double ratio;
      if (sscanf(str+1, "%lf", &ratio) != 1) return 1;
      *n *= ratio;
    }
  } else if (str[0] == '=') {
    *none = 1;
  } else {
    if (sscanf(str, "%d", n) != 1) return 1;
  }
  if (*n < 2 && !none) {
    cerr << "Error - invalid # of samples ("<<*n<<")\n";
    return 1;
  }
  return 0;
}

int NrrdResample::get_sizes(string *resampAxis, Nrrd *nrrd,
			    nrrdResampleInfo *info) {
  cerr << "NrrdResample sizes: ";
  for (int a=0; a<3; a++) {
    info->samples[a]=nrrd->axis[a].size;
    const char *str = resampAxis[a].c_str();
    int none=0;
    if (getint(str, &(info->samples[a]), &none)) { 
      cerr << "NrrdResample -- bad size.\n"; 
      return 0;
    }
    if (none) info->kernel[a]=0;
    cerr << info->samples[a];
    if (!info->kernel[a]) cerr << "=";
    cerr << " ";
  }
  cerr << endl;
  return 1;
}

void 
NrrdResample::execute()
{
  NrrdDataHandle nrrdH;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("Nrrd");
  onrrd_ = (NrrdOPort *)get_oport("Nrrd");
  if (!inrrd_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!onrrd_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if (!inrrd_->get(nrrdH))
    return;
  if (!nrrdH.get_rep()) {
    error("Error: empty Nrrd\n");
    return;
  }

  string ftype=filtertype_.get();
  string resampAxis[3];
  resampAxis[0]=resampAxis0_.get();
  resampAxis[1]=resampAxis1_.get();
  resampAxis[2]=resampAxis2_.get();
  if (last_generation_ == nrrdH->generation &&
      last_filtertype_ == ftype &&
      last_resampAxis0_ == resampAxis[0] &&
      last_resampAxis1_ == resampAxis[1] &&
      last_resampAxis2_ == resampAxis[2] &&
      last_nrrdH_.get_rep()) {
    onrrd_->send(last_nrrdH_);
    return;
  }

  nrrdResampleInfo *info = nrrdResampleInfoNew();

  Nrrd *nin = nrrdH->nrrd;
  cerr << "Resampling with a "<<ftype<<" filter."<<endl;
  nrrdKernel *kern;
  float p[NRRD_KERNEL_PARAMS_MAX];
  memset(p, 0, NRRD_KERNEL_PARAMS_MAX*sizeof(float));
  p[0]=1.0;
  if (ftype == "box") kern=nrrdKernelBox;
  else if (ftype == "tent") kern=nrrdKernelTent;
  else if (ftype == "cubicCR") { kern=nrrdKernelBCCubic; p[1]=0; p[2]=0.5; }
  else if (ftype == "cubicBS") { kern=nrrdKernelBCCubic; p[1]=1; p[2]=0; }
  else /* if (ftype == "quartic") */ { kern=nrrdKernelAQuartic; p[1]=0.0834; }
  for (int a=0; a<3; a++) {
    info->kernel[a] = kern;
    memcpy(info->param[a], p, NRRD_KERNEL_PARAMS_MAX*sizeof(float));
    info->min[a] = 0;
    info->max[a] = nin->axis[a].size-1;
  }    
  info->boundary = nrrdBoundaryBleed;
  info->type = nin->type;
  info->renormalize = AIR_TRUE;

  if (!get_sizes(resampAxis, nrrdH->nrrd, info)) return;

  last_generation_ = nrrdH->generation;
  last_filtertype_ = ftype;
  last_resampAxis0_ = resampAxis[0];
  last_resampAxis1_ = resampAxis[1];
  last_resampAxis2_ = resampAxis[2];

  NrrdData *nrrd = scinew NrrdData;
  if (nrrdSpatialResample(nrrd->nrrd=nrrdNew(), nin, info)) {
    char *err = biffGet(NRRD);
    fprintf(stderr, "NrrdResample: trouble resampling:\n%s\n", err);
    cerr << "  input Nrrd: nin->dim="<<nin->dim<<"\n";
    free(err);
  }
  nrrdResampleInfoNix(info);  
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

} // End namespace SCITeem
