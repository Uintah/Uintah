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
 *  UnuSplice.cc Replace a slice with a different nrrd. This is functionally the
 *  opposite of "slice".
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuSplice : public Module {
public:
  UnuSplice(GuiContext*);

  virtual ~UnuSplice();

  virtual void execute();

private:
  GuiInt       axis_;
  GuiInt       position_;
};


DECLARE_MAKER(UnuSplice)
UnuSplice::UnuSplice(GuiContext* ctx)
  : Module("UnuSplice", ctx, Source, "UnuNtoZ", "Teem"),
    axis_(get_ctx()->subVar("axis"), 0),
    position_(get_ctx()->subVar("position"), 0)
{
}


UnuSplice::~UnuSplice()
{
}


void
UnuSplice::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  NrrdDataHandle slice_handle;
  if (!get_input_handle("SliceNrrd", slice_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *slice = slice_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  // position could be an integer or M-<integer>
  if (!( AIR_IN_CL(0, (unsigned int) axis_.get(), nin->dim-1) )) {
    error("Axis " + to_string(axis_.get()) + " not in range [0," + to_string(nin->dim-1) + "]");
    return;
  }

  // FIX ME (ability to have M-<int>)
  
  if (nrrdSplice(nout, nin, slice, axis_.get(), position_.get())) {
    char *err = biffGetDone(NRRD);
    error(string("Error Splicing nrrd: ") + err);
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

  send_output_handle("OutputNrrd", out);
}

} // End namespace Teem


