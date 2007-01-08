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

//    File   : UnuHistax.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class UnuHistax : public Module {
public:
  UnuHistax(SCIRun::GuiContext *ctx);
  virtual ~UnuHistax();
  virtual void execute();

private:
  GuiInt       axis_;
  GuiInt       bins_;
  GuiDouble    min_;
  GuiInt       useinputmin_;
  GuiDouble    max_;
  GuiInt       useinputmax_;
  GuiString    type_;
};

DECLARE_MAKER(UnuHistax)

UnuHistax::UnuHistax(SCIRun::GuiContext *ctx) : 
  Module("UnuHistax", ctx, Filter, "UnuAtoM", "Teem"),
  axis_(get_ctx()->subVar("axis")),
  bins_(get_ctx()->subVar("bins")),
  min_(get_ctx()->subVar("min")),
  useinputmin_(get_ctx()->subVar("useinputmin")),
  max_(get_ctx()->subVar("max")),
  useinputmax_(get_ctx()->subVar("useinputmax")),
  type_(get_ctx()->subVar("type"))
{
}


UnuHistax::~UnuHistax()
{
}


void 
UnuHistax::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  NrrdRange *range = NULL;

  double min = AIR_NAN, max = AIR_NAN;
  if (!useinputmin_.get())
    min = min_.get();
  if (!useinputmax_.get())
    max = max_.get();

  nrrdRangeNew(min, max);
  nrrdRangeSafeSet(range, nin, nrrdBlind8BitRangeState);

  unsigned int type = string_to_nrrd_type(type_.get());

  if (nrrdHistoAxis(nout, nin, range, axis_.get(), bins_.get(), type)) {
    char *err = biffGetDone(NRRD);
    error(string("Error creating Histaxing nrrd: ") + err);
    free(err);
  }

  NrrdDataHandle out(scinew NrrdData(nout));

  send_output_handle("OutputNrrd", out);
}


} // End namespace SCITeem
