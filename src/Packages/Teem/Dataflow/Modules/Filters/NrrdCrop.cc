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
 *  NrrdCrop
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

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

class NrrdCrop : public Module {
  NrrdIPort* inrrd_;
  NrrdOPort* onrrd_;
  GuiInt minAxis0_;
  GuiInt maxAxis0_;
  GuiInt minAxis1_;
  GuiInt maxAxis1_;
  GuiInt minAxis2_;
  GuiInt maxAxis2_;
  GuiInt minAxis3_;
  GuiInt maxAxis3_;
  GuiInt absmaxAxis0_;
  GuiInt absmaxAxis1_;
  GuiInt absmaxAxis2_;
  GuiInt absmaxAxis3_;
  int lastmin_[4];
  int lastmax_[4];
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  int valid_data(string *minS, string *maxS, int *min, int *max, Nrrd *nrrd);
  int getint(const char *str, int *n);
  NrrdCrop(SCIRun::GuiContext *ctx);
  virtual ~NrrdCrop();
  virtual void execute();
};

} // End namespace SCITeem
using namespace SCITeem;
DECLARE_MAKER(NrrdCrop)

NrrdCrop::NrrdCrop(SCIRun::GuiContext *ctx) : 
  Module("NrrdCrop", ctx, Filter, "Filters", "Teem"), 
  minAxis0_(ctx->subVar("minAxis0")),
  maxAxis0_(ctx->subVar("maxAxis0")),
  minAxis1_(ctx->subVar("minAxis1")),
  maxAxis1_(ctx->subVar("maxAxis1")),
  minAxis2_(ctx->subVar("minAxis2")),
  maxAxis2_(ctx->subVar("maxAxis2")), 
  minAxis3_(ctx->subVar("minAxis3")),
  maxAxis3_(ctx->subVar("maxAxis3")),
  absmaxAxis0_(ctx->subVar("absmaxAxis0")),
  absmaxAxis1_(ctx->subVar("absmaxAxis1")),
  absmaxAxis2_(ctx->subVar("absmaxAxis2")),
  absmaxAxis3_(ctx->subVar("absmaxAxis3")),
  last_generation_(-1), 
  last_nrrdH_(0)
{
  for (int i = 0; i < 4; i++) {
    lastmin_[i] = -1;
    lastmax_[i] = -1;
  }
}

NrrdCrop::~NrrdCrop() {
}

void 
NrrdCrop::execute()
{
  NrrdDataHandle nrrdH;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("Nrrd");
  onrrd_ = (NrrdOPort *)get_oport("Nrrd");

  if (!inrrd_) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }
  if (!inrrd_->get(nrrdH))
    return;
  if (!nrrdH.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }
  absmaxAxis0_.reset();
  absmaxAxis1_.reset();
  absmaxAxis2_.reset();
  absmaxAxis3_.reset();
  minAxis0_.reset();
  maxAxis0_.reset();
  minAxis1_.reset();
  maxAxis1_.reset();
  minAxis2_.reset();
  maxAxis2_.reset();
  minAxis3_.reset();
  maxAxis3_.reset();
  
  if (last_generation_ != nrrdH->generation) {
    ostringstream str;
    
    absmaxAxis0_.set(nrrdH->get_tuple_axis_size() - 1);
    maxAxis0_.set(nrrdH->get_tuple_axis_size() - 1);
    absmaxAxis1_.set(nrrdH->nrrd->axis[1].size - 1);
    maxAxis1_.set(nrrdH->nrrd->axis[1].size - 1);
    if (nrrdH->nrrd->dim > 2) { 
      absmaxAxis2_.set(nrrdH->nrrd->axis[2].size - 1); 
      maxAxis2_.set(nrrdH->nrrd->axis[2].size - 1);
    } else {
      absmaxAxis2_.set(0);
      maxAxis2_.set(0);
    }
    if (nrrdH->nrrd->dim > 3) { 
      absmaxAxis3_.set(nrrdH->nrrd->axis[3].size - 1); 
      maxAxis3_.set(nrrdH->nrrd->axis[3].size - 1);
    } else {
      absmaxAxis3_.set(0);
      maxAxis3_.set(0);
    }
    
    str << id.c_str() << " set_max_vals" << endl; 
    gui->execute(str.str());
  }


  int min[4], max[4];
  min[0] = minAxis0_.get();
  max[0] = maxAxis0_.get();
  min[1] = minAxis1_.get();
  max[1] = maxAxis1_.get();
  min[2] = minAxis2_.get();
  max[2] = maxAxis2_.get();
  min[3] = minAxis3_.get();
  max[3] = maxAxis3_.get();
  
  if (last_generation_ == nrrdH->generation && last_nrrdH_.get_rep()) {
    bool same = true;
    for (int i = 0; i < 4; i++) {
      if (lastmin_[i] != min[i]) {
	same = false;
	lastmin_[i] = min[i];
      }
      if (lastmax_[i] != max[i]) {
	same = false;
	lastmax_[i] = max[i];
      }
    }
    if (same) {
      onrrd_->send(last_nrrdH_);
      return;
    }
  }
  last_generation_ = nrrdH->generation;

  Nrrd *nin = nrrdH->nrrd;
  Nrrd *nout = nrrdNew();

  // translate the tuple index into the real offsets for a tuple axis.
  int tmin, tmax;
  if (! nrrdH->get_tuple_index_info(min[0], max[0], tmin, tmax)) {
    error("Tuple index out of range");
    return;
  }
  vector<string> elems;
  nrrdH->get_tuple_indecies(elems);
  string olabel;
  for (int i = min[0]; i <= max[0]; i++) {
    if (i == min[0]) { // first one;
      olabel = elems[i];
    } else {
      olabel = olabel + "," + elems[i];
    }
  }

  min[0] = tmin;
  max[0] = tmax;
  if (nrrdCrop(nout, nin, min, max)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") + err);
    msgStream_ << "  input Nrrd: nin->dim="<<nin->dim<<"\n";
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  nout->axis[0].label = strdup(olabel.c_str());
  nrrd->copy_sci_data(*nrrdH.get_rep());
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

