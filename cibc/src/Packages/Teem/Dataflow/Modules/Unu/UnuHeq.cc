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

//    File   : UnuHeq.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class UnuHeq : public Module {
public:
  UnuHeq(SCIRun::GuiContext *ctx);
  virtual ~UnuHeq();
  virtual void execute();

private:
  GuiInt       bins_;
  GuiInt       sbins_;
  GuiDouble    amount_;
  
};

DECLARE_MAKER(UnuHeq)

UnuHeq::UnuHeq(SCIRun::GuiContext *ctx) : 
  Module("UnuHeq", ctx, Filter, "UnuAtoM", "Teem"),
  bins_(get_ctx()->subVar("bins"), 0),
  sbins_(get_ctx()->subVar("sbins"), 0),
  amount_(get_ctx()->subVar("amount"), 1.0)
{
}


UnuHeq::~UnuHeq()
{
}


void 
UnuHeq::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  if (nrrdHistoEq(nout, nin, NULL, bins_.get(), sbins_.get(), amount_.get()))
  {
    char *err = biffGetDone(NRRD);
    error(string("Error creating Heqing nrrd: ") + err);
    free(err);
  }

  NrrdDataHandle out(scinew NrrdData(nout));

  nrrdKeyValueCopy(nout, nin);

  send_output_handle("OutputNrrd", out);
}


} // End namespace SCITeem


