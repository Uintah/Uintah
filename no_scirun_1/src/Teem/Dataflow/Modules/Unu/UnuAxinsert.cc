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
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Math/MiscMath.h>
namespace SCITeem {

using namespace SCIRun;

class UnuAxinsert : public Module {
public:
  UnuAxinsert(SCIRun::GuiContext *ctx);
  virtual ~UnuAxinsert();
  virtual void execute();

private:
  GuiString       axis_;
  GuiString       label_;
};

DECLARE_MAKER(UnuAxinsert)

UnuAxinsert::UnuAxinsert(SCIRun::GuiContext *ctx) : 
  Module("UnuAxinsert", ctx, Filter, "UnuAtoM", "Teem"), 
  axis_(get_ctx()->subVar("axis")), label_(get_ctx()->subVar("label"))
{
}


UnuAxinsert::~UnuAxinsert()
{
}


void 
UnuAxinsert::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  int axis; 
  if (!string_to_int(axis_.get(), axis)) {
    axis = nin->dim;
    warning(axis_.get()+" is not a valid axis number.");
    warning("Using axis number: "+to_string(axis));
  }
  axis = Clamp(axis, 0, nin->dim);
  axis_.set(to_string(axis));
  
  if (nrrdAxesInsert(nout, nin, axis)) {
    char *err = biffGetDone(NRRD);
    error(string("Error Axinserting nrrd: ") + err);
    free(err);
  }
  
  if (strlen(label_.get().c_str())) {
    nout->axis[axis].label = airStrdup(label_.get().c_str());
  }

  NrrdDataHandle out(scinew NrrdData(nout));

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // set kind
  // Copy the axis kinds
  unsigned int offset = 0;
  for (unsigned int i=0; i<nin->dim; i++) {
    if (i == (unsigned int) axis) {
      offset = 1;
      nout->axis[i].kind = nrrdKindStub;
    }
    nout->axis[i+offset].kind = nin->axis[i].kind;
  }
  if ((unsigned int) axis == nin->dim) 
    nout->axis[axis].kind = nrrdKindStub;

  send_output_handle("OutputNrrd", out);
}


} // End namespace SCITeem


