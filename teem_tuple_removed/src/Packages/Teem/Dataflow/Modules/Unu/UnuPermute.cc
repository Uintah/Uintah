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
 *  UnuPermute
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
using std::endl;
#include <stdio.h>

namespace SCITeem {

class UnuPermute : public Module {
public:
  int valid_data(int* axes);
  UnuPermute(GuiContext *ctx);
  virtual ~UnuPermute();
  virtual void execute();
private:
  void load_gui();

  NrrdIPort           *inrrd_;
  NrrdOPort           *onrrd_;
  GuiInt               dim_;
  vector<GuiInt*>      axes_;
  vector<int>          last_axes_;
  int                  last_generation_;
  NrrdDataHandle       last_nrrdH_;
};

} // End namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(UnuPermute)

UnuPermute::UnuPermute(GuiContext *ctx) : 
  Module("UnuPermute", ctx, Filter, "Unu", "Teem"),
  dim_(ctx->subVar("dim")),
  last_generation_(-1), 
  last_nrrdH_(0)
{
  // value will be overwritten at gui side initialization.
  dim_.set(0);
}

UnuPermute::~UnuPermute() {
}

void
UnuPermute::load_gui() {
  dim_.reset();
  if (dim_.get() == 0) { return; }

  last_axes_.resize(dim_.get(), -1);

  for (int a = 0; a < dim_.get(); a++) {
    ostringstream str;
    str << "axis" << a;
    axes_.push_back(new GuiInt(ctx->subVar(str.str())));
  }
}

// check to see that the axes specified are in bounds, and not the tuple axis.
int
UnuPermute::valid_data(int* axes) {

  vector<int> exists(dim_.get(), 0);
  for (int a = 1; a < dim_.get(); a++) {
    if (axes[a] >= 1 && axes[a] < dim_.get() && !exists[axes[a]]) {
      exists[axes[a]] = 1;
    } else {
      error("Bad axis assignments!");
      return 0;
    }
  }
  return 1;
}

void 
UnuPermute::execute()
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

  if (last_generation_ != nrrdH->generation) {
    ostringstream str;
    
    if (last_generation_ != -1) {
      last_axes_.clear();
      vector<GuiInt*>::iterator iter = axes_.begin();
      while(iter != axes_.end()) {
	delete *iter;
	++iter;
      }
      axes_.clear();
      gui->execute(id.c_str() + string(" clear_axes"));
    }

    dim_.set(nrrdH->nrrd->dim);
    dim_.reset();
    load_gui();
    gui->execute(id.c_str() + string(" init_axes"));

  }
  
  if (dim_.get() == 0) { return; }

  for (int a = 0; a < dim_.get(); a++) {
    axes_[a]->reset();
  }


  bool same = true;
  for (int i = 0; i < dim_.get(); i++) {
    if (last_axes_[i] != axes_[i]->get()) {
      same = false;
      last_axes_[i] = axes_[i]->get();
    }
  }
  
  if (same && last_nrrdH_.get_rep()) {
    onrrd_->send(last_nrrdH_);
    return;
  }

  int* axp = &(last_axes_[0]);
  if (!valid_data(axp)) return;

  last_generation_ = nrrdH->generation;

  Nrrd *nin = nrrdH->nrrd;
  Nrrd *nout = nrrdNew();

  nrrdAxesPermute(nout, nin, axp);
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  nrrd->copy_sci_data(*nrrdH.get_rep());
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}

