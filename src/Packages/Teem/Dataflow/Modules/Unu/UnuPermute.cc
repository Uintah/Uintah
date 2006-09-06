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
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {
using namespace SCIRun;

class UnuPermute : public Module {
public:
  int valid_data(unsigned int* axes);
  UnuPermute(GuiContext *ctx);
  virtual ~UnuPermute();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiInt               dim_;
  GuiInt               uis_;
  vector<GuiInt*>      axes_;
  vector<unsigned int> last_axes_;
  int                  last_generation_;
  NrrdDataHandle       last_nrrdH_;
};

} // End namespace SCITeem

using namespace SCITeem;

DECLARE_MAKER(UnuPermute)

UnuPermute::UnuPermute(GuiContext *ctx) : 
  Module("UnuPermute", ctx, Filter, "UnuNtoZ", "Teem"),
  dim_(get_ctx()->subVar("dim")), uis_(get_ctx()->subVar("uis")),
  last_generation_(-1), 
  last_nrrdH_(0)
{
  // value will be overwritten at gui side initialization.
  dim_.set(0);

  for (int a = 0; a < 4; a++) 
  {
    ostringstream str;
    str << "axis" << a;
    axes_.push_back(new GuiInt(get_ctx()->subVar(str.str())));
    last_axes_.push_back(a);
  }
}


UnuPermute::~UnuPermute()
{
}


// check to see that the axes specified are in bounds
int
UnuPermute::valid_data(unsigned int* axes) {

  vector<unsigned int> exists(dim_.get(), 0);
  for (int a = 0; a < dim_.get(); a++) {
    if (axes[a] < (unsigned int) dim_.get() && !exists[axes[a]]) {
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
  update_state(NeedData);

  NrrdDataHandle nrrdH;
  if (!get_input_handle("Nrrd", nrrdH)) return;

  reset_vars();

  dim_.set(nrrdH->nrrd_->dim);

  if (dim_.get() == 0) { return; }

  bool new_dataset = (last_generation_ != nrrdH->generation);
  bool first_time = (last_generation_ == -1);

  // create any axes that might have been saved
  if (first_time) {
    uis_.reset();
    for(int i=4; i<uis_.get(); i++) {
      ostringstream str, str2;
      str << "axis" << i;
      str2 << i;
      axes_.push_back(new GuiInt(get_ctx()->subVar(str.str())));
      last_axes_.push_back(axes_[i]->get());
      get_gui()->execute(get_id().c_str() + string(" make_axis " + str2.str()));
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
      vector<GuiInt*>::iterator iter = axes_.end();
      vector<unsigned int>::iterator iter2 = last_axes_.end();
      axes_.erase(iter, iter);
      last_axes_.erase(iter2, iter2);
      get_gui()->execute(get_id().c_str() + string(" clear_axis " + str.str()));
    }
    uis_.set(nrrdH->nrrd_->dim);
  } else if ((unsigned int) uis_.get() < nrrdH->nrrd_->dim) {
    for (int i=uis_.get()-1; i< dim_.get(); i++) {
      ostringstream str, str2;
      str << "axis" << i;
      str2 << i;
      axes_.push_back(new GuiInt(get_ctx()->subVar(str.str())));
      last_axes_.push_back(i);
      get_gui()->execute(get_id().c_str() + string(" make_axis " + str2.str()));
    }
    uis_.set(nrrdH->nrrd_->dim);
  }

  for (int a = 0; a < dim_.get(); a++) {
    axes_[a]->reset();
  }

  // See if gui values have changed from last execute,
  // and set up execution values. 
  bool changed = false;

  for (int a = 0; a < dim_.get(); a++) {
    axes_[a]->reset();
  }

  for (int a = 0; a < dim_.get(); a++) {
    if (last_axes_[a] != (unsigned int) axes_[a]->get()) {
      changed = true;
      last_axes_[a] = axes_[a]->get();
    }
  }

  if (!changed && !new_dataset && last_nrrdH_.get_rep()) {
    send_output_handle("Nrrd", last_nrrdH_, true);
    return;
  }

  unsigned int* axp = &(last_axes_[0]);
  if (!valid_data(axp)) return;

  Nrrd *nin = nrrdH->nrrd_;
  Nrrd *nout = nrrdNew();

  nrrdAxesPermute(nout, nin, axp);

  last_nrrdH_ = scinew NrrdData(nout);

  send_output_handle("Nrrd", last_nrrdH_, true);
}


void 
UnuPermute::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("UnuPermute needs a minor command");
    return;
  }

  if( args[1] == "add_axis" ) 
  {
      uis_.reset();
      int i = uis_.get();
      ostringstream str, str2;
      str << "axis" << i;
      str2 << i;
      axes_.push_back(new GuiInt(get_ctx()->subVar(str.str())));
      last_axes_.push_back(i);
      get_gui()->execute(get_id().c_str() + string(" make_axis " + str2.str()));
      uis_.set(uis_.get() + 1);
  }
  else if( args[1] == "remove_axis" ) 
  {
    uis_.reset();
    int i = uis_.get()-1;
    ostringstream str;
    str << i;
    vector<GuiInt*>::iterator iter = axes_.end();
    vector<unsigned int>::iterator iter2 = last_axes_.end();
    axes_.erase(iter, iter);
    last_axes_.erase(iter2, iter2);
    get_gui()->execute(get_id().c_str() + string(" clear_axis " + str.str()));
    uis_.set(uis_.get() - 1);
  }
  else 
  {
    Module::tcl_command(args, userdata);
  }
}
