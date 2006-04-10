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
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuLut : public Module {
public:
  UnuLut(GuiContext*);

  virtual ~UnuLut();

  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdIPort*      ilut_;
  NrrdOPort*      onrrd_;

  GuiInt       rescale_;
  GuiDouble    min_;
  GuiInt       useinputmin_;
  GuiDouble    max_;
  GuiInt       useinputmax_;
  GuiString    type_;
  GuiInt       usetype_;
};


DECLARE_MAKER(UnuLut)
UnuLut::UnuLut(GuiContext* ctx)
  : Module("UnuLut", ctx, Source, "UnuAtoM", "Teem"),
    inrrd_(0), ilut_(0), onrrd_(0),
    rescale_(get_ctx()->subVar("rescale")),
    min_(get_ctx()->subVar("min")),
    useinputmin_(get_ctx()->subVar("useinputmin")),
    max_(get_ctx()->subVar("max")),
    useinputmax_(get_ctx()->subVar("useinputmax")),
    type_(get_ctx()->subVar("type")),
    usetype_(get_ctx()->subVar("usetype"))
{
}


UnuLut::~UnuLut()
{
}


void
UnuLut::execute()
{
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle lut_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  ilut_ = (NrrdIPort *)get_iport("LookupTableNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

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

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *lut = lut_handle->nrrd_;
  Nrrd *nout = nrrdNew();
  NrrdRange *range = 0;

  int rescale = rescale_.get();
  if (!( airExists(lut->axis[lut->dim - 1].min) && 
	 airExists(lut->axis[lut->dim - 1].max) )) {
    rescale = AIR_TRUE;
  }

  if (rescale) {
    double min = AIR_NAN, max = AIR_NAN;
    if (!useinputmin_.get())
      min = min_.get();
    if (!useinputmax_.get())
      max = max_.get();
    range = nrrdRangeNew(min, max);
    nrrdRangeSafeSet(range, nin, nrrdBlind8BitRangeState);
  }

  if (usetype_.get()) {
    if (nrrdApply1DLut(nout, nin, range, lut, lut->type, rescale)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Mapping Nrrd to Lookup Table: ") + err);
      free(err);
    }
  } else {
    if (nrrdApply1DLut(nout, nin, range, lut,
                       string_to_nrrd_type(type_.get()), rescale))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error Mapping Nrrd to Lookup Table: ") + err);
      free(err);
    }
  }

  NrrdDataHandle out(scinew NrrdData(nout));

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (unsigned int i=0; i<nin->dim && i<nout->dim; i++)
  {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  onrrd_->send_and_dereference(out);
}


} // End namespace Teem


