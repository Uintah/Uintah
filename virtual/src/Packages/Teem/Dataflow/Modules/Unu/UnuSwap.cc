/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

//    File   : UnuSwap.cc Interchange scan-line ordering of two axes
//    Author : Darby Van Uitert
//    Date   : April 2004

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>


namespace SCITeem {

using namespace SCIRun;

class UnuSwap : public Module {
public:
  UnuSwap(GuiContext*);

  virtual ~UnuSwap();

  virtual void execute();

private:
  GuiInt          axisA_;
  GuiInt          axisB_;
};


DECLARE_MAKER(UnuSwap)
UnuSwap::UnuSwap(GuiContext* ctx)
  : Module("UnuSwap", ctx, Source, "UnuNtoZ", "Teem"),
    axisA_(get_ctx()->subVar("axisA"), 0),
    axisB_(get_ctx()->subVar("axisB"), 1)
{
}


UnuSwap::~UnuSwap()
{
}


void
UnuSwap::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  if (nrrdAxesSwap(nout, nin, axisA_.get(), axisB_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error Swapping nrrd: ") + err);
    free(err);
  }

  NrrdDataHandle out(scinew NrrdData(nout));

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (unsigned int i=0; i<nin->dim && i<nout->dim; i++)
  {
    nout->axis[i].kind = nin->axis[i].kind;
  }
  nout->axis[axisA_.get()].kind = nin->axis[axisB_.get()].kind;
  nout->axis[axisB_.get()].kind = nin->axis[axisA_.get()].kind;

  send_output_handle("OutputNrrd", out);
}

} // End namespace Teem


