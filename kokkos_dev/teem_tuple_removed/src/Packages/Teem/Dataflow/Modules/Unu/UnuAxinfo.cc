/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  UnuAxinfo.cc:
 *
 *  Written by:
 *   Darby Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/share/share.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Teem/Core/Datatypes/NrrdData.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

namespace SCITeem {
using namespace SCIRun;

class UnuAxinfo : public Module {
public:
  UnuAxinfo(GuiContext* ctx);
  virtual ~UnuAxinfo();

  virtual void execute();

  GuiInt              axis_;
  GuiString           label_;
  GuiString           kind_;
  GuiDouble           min_;
  GuiDouble           max_;
  GuiDouble           spacing_;
private:
  int                 generation_;
};

DECLARE_MAKER(UnuAxinfo)

UnuAxinfo::UnuAxinfo(GuiContext* ctx)
  : Module("UnuAxinfo", ctx, Source, "Unu", "Teem"),
    axis_(ctx->subVar("axis")),
    label_(ctx->subVar("label")),
    kind_(ctx->subVar("kind")),
    min_(ctx->subVar("min")),
    max_(ctx->subVar("max")),
    spacing_(ctx->subVar("spacing")),
    generation_(-1)
{
}

UnuAxinfo::~UnuAxinfo(){
}


void UnuAxinfo::execute()
{
  NrrdIPort *iport = (NrrdIPort*)get_iport("Nrrd"); 
  NrrdOPort *oport = (NrrdOPort*)get_oport("Nrrd");
  
  update_state(NeedData);
  
  if (!iport) 
    {
      error("Unable to initialize iport 'Nrrd'.");
      return;
    }
  
  if (!oport) 
    {
      error("Unable to initialize oport 'Nrrd'.");
      return;
    }
  
  // The input port (with data) is required.
  NrrdDataHandle nh;
  if (!iport->get(nh)) 
    {
      return;
    }
  
  int axis = axis_.get();
  if( !nh.get_rep() || generation_ != nh->generation ) {
    if (axis >= nh->nrrd->dim) {
      error("Please specify an axis within proper range.");
      return;
    }
    
    generation_ = nh->generation;
  }
  
  reset_vars();

  Nrrd *nin = nh->nrrd;
  Nrrd *nout = nrrdNew();
  
  // copy input nrrd and modify its label, kind, min, max and spacing
  if (nrrdCopy(nout, nin)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble copying input nrrd: ") +  err);
    msgStream_ << "  input Nrrd: nin->dim=" << nin->dim << "\n";
    free(err);
    return;
  }
  
  
  if (strlen(label_.get().c_str())) {
    nout->axis[axis].label = (char*)airFree(nout->axis[axis].label);
    nout->axis[axis].label = airStrdup(const_cast<char*>(label_.get().c_str()));
    
    string kind = kind_.get();
    if (kind == "nrrdKindDomain") {
      nout->axis[axis].kind = nrrdKindDomain;
    } else if (kind == "nrrdKindScalar") {
      nout->axis[axis].kind = nrrdKindScalar;
    } else if (kind == "nrrdKind3Color") {
      nout->axis[axis].kind = nrrdKind3Color;
    } else if (kind == "nrrdKind3Vector") {
      nout->axis[axis].kind = nrrdKind3Vector;
    } else if (kind == "nrrdKind3Normal") {
      nout->axis[axis].kind = nrrdKind3Normal;
    } else if (kind == "nrrdKind3DSymTensor") {
      nout->axis[axis].kind = nrrdKind3DSymTensor;
    } else if (kind == "nrrdKind3DMaskedSymTensor") {
      nout->axis[axis].kind = nrrdKind3DMaskedSymTensor;
    } else if (kind == "nrrdKind3DTensor") {
      nout->axis[axis].kind = nrrdKind3DTensor;
    } else if (kind == "nrrdKindList") {
      nout->axis[axis].kind = nrrdKindList;
    } else if (kind == "nrrdKindStub") {
      nout->axis[axis].kind = nrrdKindStub;
    } else {
      nout->axis[axis].kind = nrrdKindUnknown;
    }
    
    if (AIR_EXISTS(min_.get())) {
      nout->axis[axis].min = min_.get();
    }
    if (AIR_EXISTS(max_.get())) {
      nout->axis[axis].max = max_.get();
    }
    if (AIR_EXISTS(spacing_.get())) {
      nout->axis[axis].spacing = spacing_.get();
    }
  }
  
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  
  NrrdDataHandle out(nrrd);
  
  // Copy the properties.
  *((PropertyManager *) out.get_rep()) =
    *((PropertyManager *) nh.get_rep());
  
  oport->send(out);

}

} // end SCITeem namespace

