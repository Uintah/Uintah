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

//    File   : UnuAxinsert.cc Add a "stub" (length 1) axis to a nrrd.
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuAxinsert : public Module {
public:
  UnuAxinsert(SCIRun::GuiContext *ctx);
  virtual ~UnuAxinsert();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          axis_;
  GuiString       label_;
};

DECLARE_MAKER(UnuAxinsert)

UnuAxinsert::UnuAxinsert(SCIRun::GuiContext *ctx) : 
  Module("UnuAxinsert", ctx, Filter, "UnuAtoM", "Teem"), 
  axis_(ctx->subVar("axis")), label_(ctx->subVar("label"))
{
}

UnuAxinsert::~UnuAxinsert() {
}

void 
UnuAxinsert::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  
  if (nrrdAxesInsert(nout, nin, axis_.get())) {
   char *err = biffGetDone(NRRD);
    error(string("Error Axinserting nrrd: ") + err);
    free(err);
  }

  if (strlen(label_.get().c_str())) {
    int axis = axis_.get();
    nout->axis[axis].label = airStrdup(label_.get().c_str());
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  // Copy the properties.
  *((PropertyManager *) out.get_rep()) =
    *((PropertyManager *) nrrd_handle.get_rep());

  // set kind
  // Copy the axis kinds
  int offset = 0;
  for (int i=0; i<nin->dim; i++) {
    if (i == axis_.get()) {
      offset = 1;
      nout->axis[i].kind = nrrdKindStub;
    }
    nout->axis[i+offset].kind = nin->axis[i].kind;
  }
  if (axis_.get() == nin->dim) 
    nout->axis[axis_.get()].kind = nrrdKindStub;

  onrrd_->send(out);

}

} // End namespace SCITeem


