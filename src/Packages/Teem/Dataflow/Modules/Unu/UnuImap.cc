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
 *  UnuImap.cc 
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class PSECORESHARE UnuImap : public Module {
public:
  UnuImap(GuiContext*);

  virtual ~UnuImap();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdIPort*      inrrd_;
  NrrdIPort*      idmap_;
  NrrdOPort*      onrrd_;

  GuiInt       length_;
  GuiInt       rescale_;
  GuiDouble    min_;
  GuiDouble    max_;
};


DECLARE_MAKER(UnuImap)
UnuImap::UnuImap(GuiContext* ctx)
  : Module("UnuImap", ctx, Source, "Unu", "Teem"),
    inrrd_(0), idmap_(0), onrrd_(0),
    length_(ctx->subVar("length")),
    rescale_(ctx->subVar("rescale")),
    min_(ctx->subVar("min")),
    max_(ctx->subVar("max"))
{
}

UnuImap::~UnuImap(){
}

void
 UnuImap::execute(){
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle dmap_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  idmap_ = (NrrdIPort *)get_iport("IrregularMapNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!idmap_) {
    error("Unable to initialize iport 'IrregularNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }

  if (!inrrd_->get(nrrd_handle))
    return;
  if (!idmap_->get(dmap_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }
  if (!dmap_handle.get_rep()) {
    error("Empty IrregularNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *dmap = dmap_handle->nrrd;
  Nrrd *nout = nrrdNew();
  Nrrd *nacl = 0;
  NrrdRange *range = 0;

  if (length_.get()) {
    nacl = nrrdNew();
    if (nrrd1DIrregAclGenerate(nacl, dmap, length_.get())) {
      char *err = biffGetDone(NRRD);
      error(string("Error generating accelerator: ") + err);
      free(err);
      return;
    }
  } else {
    nacl = NULL;
  }

  int rescale = rescale_.get();
  if (rescale) {
    range = nrrdRangeNew(min_.get(), max_.get());
    nrrdRangeSafeSet(range, nin, nrrdBlind8BitRangeState);
  }

  if (nrrdApply1DIrregMap(nout, nin, range, dmap, nacl, dmap->type, rescale)) {
    char *err = biffGetDone(NRRD);
    error(string("Error Mapping Nrrd to Lookup Table: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  // Copy the properties.
  *((PropertyManager *) out.get_rep()) =
    *((PropertyManager *) nrrd_handle.get_rep());

  // Copy the axis kinds
  for (int i=0; i<nin->dim, i<nout->dim; i++) {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  onrrd_->send(out);
}

void
 UnuImap::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem




