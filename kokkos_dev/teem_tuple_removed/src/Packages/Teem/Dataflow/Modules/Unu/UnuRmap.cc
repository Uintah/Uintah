//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
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
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class PSECORESHARE UnuRmap : public Module {
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
  GuiDouble    max_;
};


DECLARE_MAKER(UnuRmap)
UnuRmap::UnuRmap(GuiContext* ctx)
  : Module("UnuRmap", ctx, Source, "Unu", "Teem"),
    inrrd_(0), idmap_(0), onrrd_(0),
    rescale_(ctx->subVar("rescale")),
    min_(ctx->subVar("min")),
    max_(ctx->subVar("max"))
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

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!idmap_) {
    error("Unable to initialize iport 'RegularMapNrrd'.");
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
    error("Empty RegularMapNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *dmap = dmap_handle->nrrd;
  Nrrd *nout = nrrdNew();
  NrrdRange *range = 0;

  int rescale = rescale_.get();
  if (!( AIR_EXISTS(dmap->axis[dmap->dim - 1].min) && 
	 AIR_EXISTS(dmap->axis[dmap->dim - 1].max) )) {
    rescale = AIR_TRUE;
  }

  if (rescale) {
    range = nrrdRangeNew(min_.get(), max_.get());
    nrrdRangeSafeSet(range, nin, nrrdBlind8BitRangeState);
  }

  if (nrrdApply1DRegMap(nout, nin, range, dmap, dmap->type, rescale)) {
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
 UnuRmap::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


