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
 *  UnuCrop
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

class UnuCrop : public Module {
public:
  int valid_data(string *minS, string *maxS, int *min, int *max, Nrrd *nrrd);
  int getint(const char *str, int *n);
  UnuCrop(SCIRun::GuiContext *ctx);
  virtual ~UnuCrop();
  virtual void execute();
  void load_gui();
private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;
  vector<GuiInt*> mins_;
  vector<GuiInt*> maxs_;
  vector<GuiInt*> absmaxs_;
  GuiInt          num_axes_;
  vector<int>     lastmin_;
  vector<int>     lastmax_;
  int             last_generation_;
  NrrdDataHandle  last_nrrdH_;
};

} // End namespace SCITeem
using namespace SCITeem;
DECLARE_MAKER(UnuCrop)

UnuCrop::UnuCrop(SCIRun::GuiContext *ctx) : 
  Module("UnuCrop", ctx, Filter, "Unu", "Teem"), 
  num_axes_(ctx->subVar("num-axes")),
  last_generation_(-1), 
  last_nrrdH_(0)
{
  // this will get overwritten when tcl side initializes, but 
  // until then make sure it is initialized.
  num_axes_.set(0); 
  load_gui();
}

UnuCrop::~UnuCrop() {
}

void
UnuCrop::load_gui() {
  num_axes_.reset();
  if (num_axes_.get() == 0) { return; }
  
 
  lastmin_.resize(num_axes_.get(), -1);
  lastmax_.resize(num_axes_.get(), -1);  

  if (mins_.size() != num_axes_.get()) {
    for (int a = 0; a < num_axes_.get(); a++) {
      ostringstream str;
      str << "minAxis" << a;
      mins_.push_back(new GuiInt(ctx->subVar(str.str())));
      ostringstream str1;
      str1 << "maxAxis" << a;
      maxs_.push_back(new GuiInt(ctx->subVar(str1.str())));
      ostringstream str2;
      str2 << "absmaxAxis" << a;
      absmaxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
    }
  }
}

void 
UnuCrop::execute()
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
  
  num_axes_.reset();
  
  if (last_generation_ != nrrdH->generation) {
    ostringstream str;

    load_gui();

    bool do_clear = false;
    // if the dim and sizes are the same don't clear.
    if ((num_axes_.get() == nrrdH->nrrd->dim)) {
      for (int a = 0; a < num_axes_.get(); a++) {
	if (a == 0) {
	  if (absmaxs_[a]->get() != nrrdH->get_tuple_axis_size() - 1) {
	    do_clear = true;
	    break;
	  }
	} else {
	  if (absmaxs_[a]->get() != nrrdH->nrrd->axis[a].size - 1) {
	    do_clear = true;
	    break;
	  }
	}
      }
    } else {
      do_clear = true;
    }


    if (do_clear) {

      lastmin_.clear();
      lastmax_.clear();
      vector<GuiInt*>::iterator iter = mins_.begin();
      while(iter != mins_.end()) {
	delete *iter;
	++iter;
      }
      mins_.clear();
      iter = maxs_.begin();
      while(iter != maxs_.end()) {
	delete *iter;
	++iter;
      }
      maxs_.clear();
      iter = absmaxs_.begin();
      while(iter != absmaxs_.end()) {
	delete *iter;
	++iter;
      }
      absmaxs_.clear();
      gui->execute(id.c_str() + string(" clear_axes"));
      
    
      num_axes_.set(nrrdH->nrrd->dim);
      num_axes_.reset();
      load_gui();
      gui->execute(id.c_str() + string(" init_axes"));

      for (int a = 0; a < num_axes_.get(); a++) {
	maxs_[a]->reset();
      }
      for (int a = 0; a < num_axes_.get(); a++) {
	mins_[a]->set(0);
	if (a == 0) {
	  absmaxs_[a]->set(nrrdH->get_tuple_axis_size() - 1);
	} else {
	  absmaxs_[a]->set(nrrdH->nrrd->axis[a].size - 1);
	}
	maxs_[a]->reset();
	absmaxs_[a]->reset();
      }
    
      str << id.c_str() << " set_max_vals" << endl; 
      gui->execute(str.str());
    
    }
  }

  if (num_axes_.get() == 0) { return; }
  
  for (int a = 0; a < num_axes_.get(); a++) {
    mins_[a]->reset();
    maxs_[a]->reset();
    absmaxs_[a]->reset();
  }



  if (last_generation_ == nrrdH->generation && last_nrrdH_.get_rep()) {
    bool same = true;
    for (int i = 0; i < num_axes_.get(); i++) {
      if (lastmin_[i] != mins_[i]->get()) {
	same = false;
	lastmin_[i] = mins_[i]->get();
      }
      if (lastmax_[i] != maxs_[i]->get()) {
	same = false;
	lastmax_[i] = maxs_[i]->get();
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
  if (! nrrdH->get_tuple_index_info(mins_[0]->get(), maxs_[0]->get(), 
				    tmin, tmax)) {
    error("Tuple index out of range");
    return;
  }
  vector<string> elems;
  nrrdH->get_tuple_indecies(elems);
  string olabel;
  for (int i = mins_[0]->get(); i <= maxs_[0]->get(); i++) {
    if (i == mins_[0]->get()) { // first one;
      olabel = elems[i];
    } else {
      olabel = olabel + "," + elems[i];
    }
  }

  int *min = scinew int[num_axes_.get()];
  int *max = scinew int[num_axes_.get()];

  for(int i = 1; i <  num_axes_.get(); i++) {
    min[i] = mins_[i]->get();
    max[i] = maxs_[i]->get();
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

  delete min;
  delete max;
}
