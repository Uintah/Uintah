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

//    File   : UnuJoin.cc
//    Author : Martin Cole
//    Date   : Wed Jan 15 10:59:57 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {
using namespace SCIRun;

class UnuJoin : public Module {
public:
  UnuJoin(SCIRun::GuiContext *ctx);
  virtual ~UnuJoin();

  virtual void execute();

private:
  NrrdOPort* onrrd_;

  NrrdDataHandle    onrrd_handle_;  //! the cached output nrrd handle.
  vector<int>       in_generation_; //! all input generation nums.
  int               onrrd_type_;    //! target type for output nrrd.

  GuiInt            join_axis_;
  GuiInt            incr_dim_;
  GuiInt            dim_;
  int               old_axis_;
  int               old_incr_dim_;
};

} // End namespace SCITeem
using namespace SCITeem;
DECLARE_MAKER(UnuJoin)

UnuJoin::UnuJoin(SCIRun::GuiContext *ctx) : 
  Module("UnuJoin", ctx, Filter, "UnuAtoM", "Teem"), 
  onrrd_(0),
  onrrd_handle_(0),
  in_generation_(0),
  onrrd_type_(nrrdTypeLast),
  join_axis_(ctx->subVar("join-axis")),
  incr_dim_(ctx->subVar("incr-dim")),
  dim_(ctx->subVar("dim")),
  old_axis_(0),
  old_incr_dim_(0)
{
}

UnuJoin::~UnuJoin() {
}

void 
UnuJoin::execute()
{
  if (! onrrd_) {
    onrrd_ = (NrrdOPort *)get_oport("JoinedNrrd");
  }
  port_range_type range = get_iports("Nrrds");
  if (range.first == range.second) { return; }

  unsigned int i = 0;
  vector<NrrdDataHandle> nrrds;
  bool do_join = false;
  int max_dim = 0;
  port_map_type::iterator pi = range.first;
  while (pi != range.second)
  {
    NrrdIPort *inrrd = (NrrdIPort *)get_iport(pi->second);
    NrrdDataHandle nrrd;
    
    if (inrrd->get(nrrd) && nrrd.get_rep()) {
      // check to see if we need to do the join or can output the cached onrrd.
      if (in_generation_.size() <= i) {
	// this is a new input, never been joined.
	do_join = true;
	in_generation_.push_back(nrrd->generation);
	onrrd_type_ = nrrdTypeLast;
      } else if (in_generation_[i] != nrrd->generation) {
	// different input than last execution
	do_join = true;
	in_generation_[i] = nrrd->generation;
	onrrd_type_ = nrrdTypeLast;
      }

      // the output nrrd must be of one type, so find the type that accomodates
      // all of the nrrds we have as input.
      if (onrrd_type_ == nrrdTypeLast) {
	// first time this value is set
	onrrd_type_ = nrrd->nrrd->type;
      }
      if ((onrrd_type_ != nrrd->nrrd->type) && 
	  (onrrd_type_ != nrrdTypeDouble)) {
	//! promote to the biggest type
	if (nrrdTypeSize[nrrd->nrrd->type] > nrrdTypeSize[onrrd_type_]){
	  onrrd_type_ = nrrd->nrrd->type;
	}
      }
      
      if (nrrd->nrrd->dim > max_dim) { max_dim = nrrd->nrrd->dim; }
      nrrds.push_back(nrrd);
    }
    ++pi; ++i;
  }

  dim_.reset();
  if (max_dim != dim_.get()) {    
    dim_.set(max_dim);
    dim_.reset();
  }

  // re-join if old axis is different from new
  // axis or incr_dim has changed
  if (old_axis_ != join_axis_.get()) {
    do_join = true;
    old_axis_ = join_axis_.get();
  }
  if (old_incr_dim_ != incr_dim_.get()) {
    do_join = true;
    old_incr_dim_ = incr_dim_.get();
  }
  
  
  vector<Nrrd*> arr(nrrds.size());

  if (do_join) {
    NrrdData *onrrd = new NrrdData();
    int i = 0;
    string new_label("");
    vector<NrrdDataHandle>::iterator iter = nrrds.begin();
    while(iter != nrrds.end()) {
      NrrdDataHandle nh = *iter;
      ++iter;

      NrrdData* cur_nrrd = nh.get_rep();
      // does it need conversion to the bigger type?
      if (cur_nrrd->nrrd->type != onrrd_type_) {
	Nrrd* new_nrrd = nrrdNew();
	if (nrrdConvert(new_nrrd, cur_nrrd->nrrd, onrrd_type_)) {
	  char *err = biffGetDone(NRRD);
	  error(string("Conversion Error: ") +  err);
	  free(err);
	  return;
	}
	arr[i] = new_nrrd;
      } else {
	arr[i] = cur_nrrd->nrrd;
      }
      ++i;
    }
    
    join_axis_.reset();
    incr_dim_.reset();

    onrrd->nrrd = nrrdNew();
    if (nrrdJoin(onrrd->nrrd, &arr[0], nrrds.size(),
		 join_axis_.get(), incr_dim_.get())) {
      char *err = biffGetDone(NRRD);
      error(string("Join Error: ") +  err);
      free(err);
      return;
    }

    onrrd_handle_ = onrrd;
  }

  onrrd_->send(onrrd_handle_);
}

