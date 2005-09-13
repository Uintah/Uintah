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

//    File   : UnuProject.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class UnuProject : public Module {
public:
  UnuProject(SCIRun::GuiContext *ctx);
  virtual ~UnuProject();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt       axis_;
  GuiInt    measure_;
  int       old_generation_;
  int       old_axis_;
  int       old_measure_;
  NrrdDataHandle last_nrrdH_;
};

DECLARE_MAKER(UnuProject)

UnuProject::UnuProject(SCIRun::GuiContext *ctx) : 
  Module("UnuProject", ctx, Filter, "UnuNtoZ", "Teem"), 
  axis_(ctx->subVar("axis")),
  measure_(ctx->subVar("measure")),
  old_generation_(-1), old_axis_(-1), old_measure_(-1),
  last_nrrdH_(0)
{
}

UnuProject::~UnuProject() {
}

void 
UnuProject::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  onrrd_ = (NrrdOPort *)get_oport("nout");

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  reset_vars();
  
  bool need_execute = false;
  if (old_axis_ != axis_.get() || 
      old_measure_ != measure_.get() ||
      old_generation_ != nrrd_handle->generation) {
    old_axis_ = axis_.get();
    old_measure_ = measure_.get();
    old_generation_ = nrrd_handle->generation;
    need_execute = true;
  }

  if (need_execute) {
    Nrrd *nin = nrrd_handle->nrrd;
    Nrrd *nout = nrrdNew();
    
    if (nrrdProject(nout, nin, axis_.get(), measure_.get(), nrrdTypeDefault)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Projecting nrrd: ") + err);
      free(err);
    }
    
    NrrdData *nrrd = scinew NrrdData;
    nrrd->nrrd = nout;
    //nrrd->copy_sci_data(*nrrd_handle.get_rep());
    last_nrrdH_ = nrrd;
  }
  onrrd_->send(last_nrrdH_);
}

} // End namespace SCITeem
