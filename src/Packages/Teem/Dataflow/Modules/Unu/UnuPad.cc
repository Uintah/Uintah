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
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

#include <iostream>
#include <stdio.h>

using std::endl;

namespace SCITeem {
using namespace SCIRun;

class UnuPad : public Module {
public:
  int valid_data(string *minS, string *maxS, int *min, int *max, Nrrd *nrrd);
  int getint(const char *str, int *n);
  UnuPad(GuiContext *ctx);
  virtual ~UnuPad();
  virtual void execute();
private:
  void load_gui();

  GuiString           pad_style_;
  GuiDouble           pad_value_;

  string              last_pad_style_;
  double              last_pad_value_;

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
  Module("UnuPad", ctx, Filter, "UnuNtoZ", "Teem"),
  pad_style_(get_ctx()->subVar("pad-style"), "Bleed"),
  pad_value_(get_ctx()->subVar("pad-value"), 0.0),
  dim_(get_ctx()->subVar("dim"), 0),
  last_generation_(-1), 
  last_nrrdH_(0)
{
  // value will be overwritten at gui side initialization.
  dim_.set(0);
}


UnuPad::~UnuPad()
{
}


void
UnuPad::load_gui()
{
  dim_.reset();
  if (dim_.get() == 0) { return; }

  last_mins_.resize(dim_.get(), -1);
  last_maxs_.resize(dim_.get(), -1);  
  for (int a = 0; a < dim_.get(); a++) {
    ostringstream str;
    str << "minAxis" << a;
    mins_.push_back(new GuiInt(get_ctx()->subVar(str.str())));
    ostringstream str1;
    str1 << "maxAxis" << a;
    maxs_.push_back(new GuiInt(get_ctx()->subVar(str1.str())));
  }
}


void 
UnuPad::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrdH;
  if (!get_input_handle("Nrrd", nrrdH)) return;

  reset_vars();

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
      get_gui()->execute(get_id().c_str() + string(" clear_axes"));
    }

    last_generation_ = nrrdH->generation;
    dim_.set(nrrdH->nrrd_->dim);
    dim_.reset();
    load_gui();
    get_gui()->execute(get_id().c_str() + string(" init_axes"));
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

  if( last_pad_style_ != pad_style_.get() ||
      last_pad_value_ != pad_value_.get() )
  {
    last_pad_style_ = pad_style_.get();
    last_pad_value_ = pad_value_.get();

    changed = true;
  }

  vector<ptrdiff_t> min(dim_.get()), max(dim_.get());
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
    max[a] = (nrrdH->nrrd_->axis[a].size - 1) + maxs_[a]->get();  

    if (nrrdKindSize(nrrdH->nrrd_->axis[a].kind) > 1 && (min[a] !=0 || max[a] != 0)) {
      warning("Trying to pad axis " + to_string(a) + " which does not have a kind of nrrdKindDomain or nrrdKindUnknown");
    }
  }

  if ( changed || !last_nrrdH_.get_rep())
  {
    Nrrd *nin = nrrdH->nrrd_;
    Nrrd *nout = nrrdNew();
    ptrdiff_t *minp = &(min[0]);
    ptrdiff_t *maxp = &(max[0]);

    if( (last_pad_style_ == "Bleed" &&
	 nrrdPad_nva(nout, nin, minp, maxp, nrrdBoundaryBleed, 0)) ||
	(last_pad_style_ == "Wrap" &&
	 nrrdPad_nva(nout, nin, minp, maxp, nrrdBoundaryWrap, 0)) ||
	(last_pad_style_ == "Pad" &&
	 nrrdPad_nva(nout, nin, minp, maxp, nrrdBoundaryPad, last_pad_value_)) )
    {
	char *err = biffGetDone(NRRD);
	error(string("Trouble resampling: ") + err);
	msg_stream_ << "  input Nrrd: nin->dim="<<nin->dim<<"\n";
	free(err);
    }

    last_nrrdH_ = scinew NrrdData(nout);

    // Copy the properies, kinds, and labels.
    last_nrrdH_->copy_properties(nrrdH.get_rep());

    for( unsigned int i=0; i<nin->dim; i++ )
    {
      nout->axis[i].kind  = nin->axis[i].kind;
      nout->axis[i].label = airStrdup(nin->axis[i].label);
    }

    if( nout->axis[0].size == 3)
      nout->axis[0].kind = nrrdKind3Vector;
    else if( nout->axis[0].size == 6 )
      nout->axis[0].kind = nrrdKind3DSymMatrix;
    else
      nout->axis[0].kind = nrrdKindDomain;
  }

  send_output_handle("Nrrd", last_nrrdH_, true);
}

