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
 *  UnuPad
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
#include <stdio.h>

using std::endl;

namespace SCITeem {

class UnuPad : public Module {
public:
  int valid_data(string *minS, string *maxS, int *min, int *max, Nrrd *nrrd);
  int getint(const char *str, int *n);
  UnuPad(GuiContext *ctx);
  virtual ~UnuPad();
  virtual void execute();
private:
  void load_gui();
  
  NrrdIPort          *inrrd_;
  NrrdOPort          *onrrd_;
  vector<GuiInt*>     mins_;
  vector<int>         last_mins_;
  vector<GuiInt*>     maxs_;
  vector<int>         last_maxs_;
  GuiInt              dim_;
  int                 last_generation_;
  NrrdDataHandle      last_nrrdH_;
};

} // End namespace SCITeem

using namespace SCITeem;

DECLARE_MAKER(UnuPad)

UnuPad::UnuPad(GuiContext *ctx) : 
  Module("UnuPad", ctx, Filter, "Unu", "Teem"),
  dim_(ctx->subVar("dim")),
  last_generation_(-1), 
  last_nrrdH_(0)
{
  // value will be overwritten at gui side initialization.
  dim_.set(0);

  
}

UnuPad::~UnuPad() {
}

void
UnuPad::load_gui() {
  dim_.reset();
  if (dim_.get() == 0) { return; }

  last_mins_.resize(dim_.get(), -1);
  last_maxs_.resize(dim_.get(), -1);  
  for (int a = 0; a < dim_.get(); a++) {
    ostringstream str;
    str << "minAxis" << a;
    mins_.push_back(new GuiInt(ctx->subVar(str.str())));
    ostringstream str1;
    str1 << "maxAxis" << a;
    maxs_.push_back(new GuiInt(ctx->subVar(str1.str())));
  }
}

void 
UnuPad::execute()
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
  dim_.reset();

  // test for new input
  if (last_generation_ != nrrdH->generation) {

    if (last_generation_ != -1) {
      last_mins_.clear();
      last_maxs_.clear();
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
      gui->execute(id.c_str() + string(" clear_axes"));
    }

    last_generation_ = nrrdH->generation;
    dim_.set(nrrdH->nrrd->dim);
    dim_.reset();
    load_gui();
    gui->execute(id.c_str() + string(" init_axes"));

  }

  dim_.reset();
  if (dim_.get() == 0) { return; }

  for (int a = 0; a < dim_.get(); a++) {
    mins_[a]->reset();
    maxs_[a]->reset();
  }
  
  // See if gui values have changed from last execute,
  // and set up execution values. 
  bool changed = false;
  vector<int> min(dim_.get()), max(dim_.get());
  for (int a = 0; a < dim_.get(); a++) {
    if (last_mins_[a] != mins_[a]->get()) {
      changed = true;
      last_mins_[a] = mins_[a]->get();
    }
    if (last_maxs_[a] != maxs_[a]->get()) {
      changed = true;
      last_maxs_[a] = maxs_[a]->get();
    }
    min[a] = 0 - mins_[a]->get();
    max[a] = (nrrdH->nrrd->axis[a].size - 1) + maxs_[a]->get();  
  }

  if (! changed) { 
    // send old nrrd
    onrrd_->send(last_nrrdH_);
    return;
  }

  Nrrd *nin = nrrdH->nrrd;
  Nrrd *nout = nrrdNew();
  int *minp = &(min[0]);
  int *maxp = &(max[0]);

  if (nrrdPad(nout, nin, minp, maxp, nrrdBoundaryBleed)) 
  {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") + err);
    msgStream_ << "  input Nrrd: nin->dim="<<nin->dim<<"\n";
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  nrrd->copy_sci_data(*nrrdH.get_rep());
  last_nrrdH_ = nrrd;
  onrrd_->send(last_nrrdH_);
}
