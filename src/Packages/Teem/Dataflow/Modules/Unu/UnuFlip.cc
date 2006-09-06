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

//    File   : UnuFlip.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuFlip : public Module {
public:
  UnuFlip(SCIRun::GuiContext *ctx);
  virtual ~UnuFlip();
  virtual void execute();

private:
  GuiInt          axis_;
};

DECLARE_MAKER(UnuFlip)

UnuFlip::UnuFlip(SCIRun::GuiContext *ctx) : 
  Module("UnuFlip", ctx, Filter, "UnuAtoM", "Teem"), 
  axis_(get_ctx()->subVar("axis"), 0)
{
}


UnuFlip::~UnuFlip()
{
}


void 
UnuFlip::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();
  
  if (axis_.get() >= (int)nin->dim)
  {
    error("Axis " + to_string(axis_.get()) +
          " out of bounds [0," + to_string(nin->dim-1));
    return;
  }

  if (nrrdFlip(nout, nin, axis_.get()))
  {
   char *err = biffGetDone(NRRD);
    error(string("Error Flipping nrrd: ") + err);
    free(err);
  }

  // set min/max of flipped axis to be what input nrrd's was
  nout->axis[axis_.get()].min = nin->axis[axis_.get()].min;
  nout->axis[axis_.get()].max = nin->axis[axis_.get()].max;

  NrrdDataHandle out(scinew NrrdData(nout));

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (size_t i=0; i<nin->dim; i++)
  {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  send_output_handle("OutputNrrd", out);
}


} // End namespace SCITeem
