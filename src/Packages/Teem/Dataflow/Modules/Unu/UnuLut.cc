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
 *  UnuLut.cc 
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

class PSECORESHARE UnuLut : public Module {
public:
  UnuLut(GuiContext*);

  virtual ~UnuLut();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdIPort*      inrrd_;
  NrrdIPort*      ilut_;
  NrrdOPort*      onrrd_;

  GuiInt       rescale_;
  GuiDouble    min_;
  GuiDouble    max_;
};


DECLARE_MAKER(UnuLut)
UnuLut::UnuLut(GuiContext* ctx)
  : Module("UnuLut", ctx, Source, "Unu", "Teem"),
    inrrd_(0), ilut_(0), onrrd_(0),
    rescale_(ctx->subVar("rescale")),
    min_(ctx->subVar("min")),
    max_(ctx->subVar("max"))
{
}

UnuLut::~UnuLut(){
}

void
 UnuLut::execute(){
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle lut_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  ilut_ = (NrrdIPort *)get_iport("LookupTableNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!ilut_) {
    error("Unable to initialize iport 'LookupTableNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }

  if (!inrrd_->get(nrrd_handle))
    return;
  if (!ilut_->get(lut_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }
  if (!lut_handle.get_rep()) {
    error("Empty LookupTableNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *lut = lut_handle->nrrd;
  Nrrd *nout = nrrdNew();
  NrrdRange *range;

  int rescale = rescale_.get();
  if (!( AIR_EXISTS(lut->axis[lut->dim - 1].min) && 
	 AIR_EXISTS(lut->axis[lut->dim - 1].max) )) {
    rescale = AIR_TRUE;
  }

  if (rescale) {
    range = nrrdRangeNew(min_.get(), max_.get());
    nrrdRangeSafeSet(range, nin, nrrdBlind8BitRangeState);
  }

  if (nrrdApply1DLut(nout, nin, range, lut, nin->type, rescale)) {
    char *err = biffGetDone(NRRD);
    error(string("Error Mapping Nrrd to Lookup Table: ") + err);
    free(err);
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
 UnuLut::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


