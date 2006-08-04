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

//    File   : UnuAxdelete.cc Remove one or more singleton axes from a nrrd.
//    Author : Darby Van Uitert
//    Date   : April 2004

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuAxdelete : public Module {
public:
  UnuAxdelete(SCIRun::GuiContext *ctx);
  virtual ~UnuAxdelete();
  virtual void execute();

private:
  GuiInt          axis_;
};

DECLARE_MAKER(UnuAxdelete)

UnuAxdelete::UnuAxdelete(SCIRun::GuiContext *ctx) : 
  Module("UnuAxdelete", ctx, Filter, "UnuAtoM", "Teem"), 
  axis_(get_ctx()->subVar("axis"), 0)
{
}


UnuAxdelete::~UnuAxdelete()
{
}


void 
UnuAxdelete::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  const int axis = axis_.get();
  if (axis < 0)
  {
    Nrrd *ntmp = nrrdNew();
    if (nrrdCopy(nout, nin))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error copying axis: ") + err);
      free(err);
    }
    unsigned int a;
    for (a=0; a<nout->dim && nout->axis[a].size > 1; a++);
    while (a < nout->dim)
    {
      if (nrrdAxesDelete(ntmp, nout, a) || nrrdCopy(nout, ntmp))
      {
	char *err = biffGetDone(NRRD);
	error(string("Error Copying deleting axis: ") + err);
	free(err);
      }
      for (a=0; a<nout->dim && nout->axis[a].size > 1; a++);
    }
    NrrdDataHandle out(scinew NrrdData(nout));
    
    // Copy the properties.
    out->copy_properties(nrrd_handle.get_rep());
    
    send_output_handle("OutputNrrd", out);
  }
  else
  {
    if (nrrdAxesDelete(nout, nin, axis))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error Axdeleting nrrd: ") + err);
      free(err);
    }

    NrrdDataHandle out(scinew NrrdData(nout));
    
    // Copy the properties.
    out->copy_properties(nrrd_handle.get_rep());
    
    // set kind
    // Copy the axis kinds
    int offset = 0;
    for (unsigned int i=0; i<nin->dim; i++)
    {
      if (i == (unsigned int)axis)
      {
	offset = 1;
      }
      else
      {
	nout->axis[i-offset].kind = nin->axis[i].kind;
      }
    }

    send_output_handle("OutputNrrd", out);
  }
}


} // End namespace SCITeem


