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
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

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
  : Module("UnuAxinfo", ctx, Source, "UnuAtoM", "Teem"),
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
    if (nout->axis[axis].label) { airFree(nout->axis[axis].label); }
    nout->axis[axis].label = airStrdup(label_.get().c_str());
    
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
    } else if (kind == "nrrdKind3DSymMatrix") {
      nout->axis[axis].kind = nrrdKind3DSymMatrix;
    } else if (kind == "nrrdKind3DMaskedSymMatrix") {
      nout->axis[axis].kind = nrrdKind3DMaskedSymMatrix;
    } else if (kind == "nrrdKind3DMatrix") {
      nout->axis[axis].kind = nrrdKind3DMatrix;
    } else if (kind == "nrrdKindList") {
      nout->axis[axis].kind = nrrdKindList;
    } else if (kind == "nrrdKindStub") {
      nout->axis[axis].kind = nrrdKindStub;
    } else {
      nout->axis[axis].kind = nrrdKindUnknown;
    }
    
    if (airExists(min_.get())) {
      nout->axis[axis].min = min_.get();
    }
    if (airExists(max_.get())) {
      nout->axis[axis].max = max_.get();
    }
    if (airExists(spacing_.get())) {
      nout->axis[axis].spacing = spacing_.get();
    }
  }
  
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  
  NrrdDataHandle out(nrrd);
  
  // Copy the properties.
  out->copy_properties(nh.get_rep());
  
  oport->send(out);

}

} // end SCITeem namespace

