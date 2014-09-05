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
 *  UnuRmap.cc 
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuRmap : public Module {
public:
  UnuRmap(GuiContext*);

  virtual ~UnuRmap();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdIPort*      inrrd_;
  NrrdIPort*      idmap_;
  NrrdOPort*      onrrd_;

  GuiInt       rescale_;
  GuiDouble    min_;
  GuiInt       useinputmin_;
  GuiDouble    max_;
  GuiInt       useinputmax_;
  GuiString    type_;
  GuiInt       usetype_;

  unsigned int get_type(string type);
};


DECLARE_MAKER(UnuRmap)
UnuRmap::UnuRmap(GuiContext* ctx)
  : Module("UnuRmap", ctx, Source, "UnuNtoZ", "Teem"),
    inrrd_(0), idmap_(0), onrrd_(0),
    rescale_(ctx->subVar("rescale")),
    min_(ctx->subVar("min")),
    useinputmin_(ctx->subVar("useinputmin")),
    max_(ctx->subVar("max")),
    useinputmax_(ctx->subVar("useinputmax")),
    type_(ctx->subVar("type")),
    usetype_(ctx->subVar("usetype"))
{
}

UnuRmap::~UnuRmap(){
}

void
 UnuRmap::execute(){
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle dmap_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  idmap_ = (NrrdIPort *)get_iport("RegularMapNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle))
    return;
  if (!idmap_->get(dmap_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }
  if (!dmap_handle.get_rep()) {
    error("Empty RegularMapNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *dmap = dmap_handle->nrrd;
  Nrrd *nout = nrrdNew();
  NrrdRange *range = NULL;

  int rescale = rescale_.get();
  if (!( airExists(dmap->axis[dmap->dim - 1].min) && 
	 airExists(dmap->axis[dmap->dim - 1].max) )) {
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
    if (nrrdApply1DRegMap(nout, nin, range, dmap, dmap->type, rescale)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Mapping Nrrd to Lookup Table: ") + err);
      free(err);
    }
  } else {
    if (nrrdApply1DRegMap(nout, nin, range, dmap, get_type(type_.get()), rescale)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Mapping Nrrd to Lookup Table: ") + err);
      free(err);
    }
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (int i=0; i<nin->dim && i<nout->dim; i++)
  {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  onrrd_->send(out);
}

void
 UnuRmap::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

unsigned int
UnuRmap::get_type(string type) {
  if (type == "nrrdTypeChar") 
    return nrrdTypeChar;
  else if (type == "nrrdTypeUChar")  
    return nrrdTypeUChar;
  else if (type == "nrrdTypeShort")  
    return nrrdTypeShort;
  else if (type == "nrrdTypeUShort") 
    return nrrdTypeUShort;
  else if (type == "nrrdTypeInt")  
    return nrrdTypeInt;
  else if (type == "nrrdTypeUInt")   
    return nrrdTypeUInt;
  else if (type == "nrrdTypeFloat") 
    return nrrdTypeFloat;
  else if (type == "nrrdTypeDouble")  
    return nrrdTypeDouble;
  else    
    return nrrdTypeUInt;
}

} // End namespace Teem


