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
 *  UnuCCfind.cc 
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuCCfind : public Module {
public:
  UnuCCfind(GuiContext*);

  virtual ~UnuCCfind();

  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;
  
  GuiInt          connectivity_;
  GuiString       type_;
  GuiInt          usetype_;
};


DECLARE_MAKER(UnuCCfind)
UnuCCfind::UnuCCfind(GuiContext* ctx)
  : Module("UnuCCfind", ctx, Source, "UnuAtoM", "Teem"),
    inrrd_(0), onrrd_(0),
    connectivity_(get_ctx()->subVar("connectivity")),
    type_(get_ctx()->subVar("type")),
    usetype_(get_ctx()->subVar("usetype"))
{
}

UnuCCfind::~UnuCCfind(){
}

void
UnuCCfind::execute(){
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  if (!usetype_.get()) {
    if (nrrdCCFind(nout, NULL, nin,
                   string_to_nrrd_type(type_.get()), connectivity_.get())) {
      char *err = biffGetDone(NRRD);
      error(string("Error performing CC find nrrd: ") + err);
      free(err);
    }
  } else {
    if (nrrdCCFind(nout, NULL, nin, nrrdTypeDefault, connectivity_.get())) {
      char *err = biffGetDone(NRRD);
      error(string("Error performing CC find nrrd: ") + err);
      free(err);
    }
  }

  NrrdDataHandle out(scinew NrrdData(nout));

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (unsigned int i=0; i<nin->dim && i<nout->dim; i++) {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  onrrd_->send_and_dereference(out);
}


} // End namespace Teem




