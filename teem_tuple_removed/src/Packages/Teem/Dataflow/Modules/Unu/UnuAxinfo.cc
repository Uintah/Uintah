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

#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCITeem {
using namespace SCIRun;

using std::endl;
using std::pair;

class UnuAxinfo : public Module {
public:
  UnuAxinfo(GuiContext* ctx);
  virtual ~UnuAxinfo();

  void load_gui();
  void clear_vals();
  virtual void execute();

  GuiInt              dimension_;
  GuiString           type_;
  GuiString           label0_;
  GuiInt              gui_initialized_;
  vector<GuiString*>  label_;
  vector<GuiString*>  kind_;
  vector<GuiString*>  center_;
  vector<GuiDouble*>  size_;
  vector<GuiDouble*>  min_;
  vector<GuiDouble*>  max_;
  vector<GuiDouble*>  spacing_;

private:
  int                 generation_;
  int                 max_vectors_;
};

DECLARE_MAKER(UnuAxinfo)

UnuAxinfo::UnuAxinfo(GuiContext* ctx)
  : Module("UnuAxinfo", ctx, Source, "Unu", "Teem"),
    dimension_(ctx->subVar("dimension")),
    type_(ctx->subVar("type")),
    label0_(ctx->subVar("label0")),
    gui_initialized_(ctx->subVar("initialized")),
    generation_(-1),
    max_vectors_(0)
{
  dimension_.set(0);
  load_gui();
}

UnuAxinfo::~UnuAxinfo(){
}


// Build up vectors of GuiVars and connect
// with corresponding tcl vars
void UnuAxinfo::load_gui() {
  dimension_.reset();
  if (dimension_.get() == 0) { return; }
  
  if(max_vectors_ != dimension_.get()) {
    for(int a = max_vectors_; a <= dimension_.get(); a++) {
      ostringstream lab, kind, cntr, sz, min, max, spac;
      lab << "label" << a;
      label_.push_back(new GuiString(ctx->subVar(lab.str())));
      kind << "kind" << a;
      kind_.push_back(new GuiString(ctx->subVar(kind.str())));
      cntr << "center" << a;
      center_.push_back(new GuiString(ctx->subVar(cntr.str())));
      sz << "size" << a;
      size_.push_back(new GuiDouble(ctx->subVar(sz.str())));
      min << "min" << a;
      min_.push_back(new GuiDouble(ctx->subVar(min.str())));
      max << "max" << a;
      max_.push_back(new GuiDouble(ctx->subVar(max.str())));
      spac << "spacing" << a;
      spacing_.push_back(new GuiDouble(ctx->subVar(spac.str())));
      
      max_vectors_++;
    }
  }
}

void UnuAxinfo::clear_vals() 
{
  gui->execute(id.c_str() + string(" clear_axes"));
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
  if (!iport->get(nh) || !nh.get_rep())
  {
    clear_vals();
    generation_ = -1;
    return;
  }

  dimension_.reset();
  bool do_clear = false;
  bool sizes_same = true;

  load_gui();
  // don't clear/reset if sizes that are saved are the same as nrrd's
  if(dimension_.get() == nh->nrrd->dim) {
    for(int a = 0; a < dimension_.get(); a++) {
      if(size_[a]->get() != nh->nrrd->axis[a].size) {
	sizes_same = false;
	break;
      }
    }
  } else {
    sizes_same = false;
  }
  
  if (generation_ != nh.get_rep()->generation && !sizes_same) 
  {
    do_clear = true;
    generation_ = nh->generation;
  }
  
  if (do_clear) {
    // delete the guivars  in the vectors and then clear
    // all of them
    vector<GuiString*>::iterator iter1 = label_.begin();
    while(iter1 != label_.end()) {
      delete *iter1;
      ++iter1;
    }
    label_.clear();
    iter1 = kind_.begin();
    while(iter1 != kind_.end()) {
      delete *iter1;
      ++iter1;
    }
    kind_.clear();
    
    iter1 = center_.begin();
    while(iter1 != center_.end()) {
      delete *iter1;
      ++iter1;
    }
    center_.clear();
    vector<GuiDouble*>::iterator iter2 = size_.begin();
    while(iter2 != size_.end()) {
      delete *iter2;
      ++iter2;
    } 
    size_.clear();
    iter2 = min_.begin();
    while(iter2 != min_.end()) {
      delete *iter2;
      ++iter2;
    } 
    min_.clear();
    iter2 = max_.begin();
    while(iter2 != max_.end()) {
      delete *iter2;
      ++iter2;
    } 
    max_.clear();
    iter2 = spacing_.begin();
    while(iter2 != spacing_.end()) {
      delete *iter2;
      ++iter2;
    } 
    spacing_.clear();
    max_vectors_ = 0;
    
    gui->execute(id.c_str() + string(" clear_axes"));
    
    dimension_.set(nh->nrrd->dim);
    dimension_.reset();
    
    load_gui();
    
    gui->execute(id.c_str() + string(" init_axes"));

    gui_initialized_.set(1);
    
    // set nrrd info to be like the nh->nrrd's
    // because this is new input
    switch (nh->nrrd->type) {
    case nrrdTypeChar :  
      type_.set("char");
      break;
    case nrrdTypeUChar : 
      type_.set("unsigned char");
      break;
    case nrrdTypeShort : 
      type_.set("short");
      break;
    case nrrdTypeUShort :
      type_.set("unsigned short");
      break;
    case nrrdTypeInt : 
      type_.set("int");
      break;
    case nrrdTypeUInt :  
      type_.set("unsigned int");
      break;
    case nrrdTypeLLong : 
      type_.set("long long");
      break;
    case nrrdTypeULLong :
      type_.set("unsigned long long");
      break;
    case nrrdTypeFloat :
      type_.set("float");
      break;
    case nrrdTypeDouble :
      type_.set("double");
      break;
    }
    for(int a = 0; a < dimension_.get(); a++) {
      if (nh->nrrd->axis[a].label == NULL || string(nh->nrrd->axis[a].label).length() == 0) {
	label_[a]->set("---");
	nh->nrrd->axis[a].label = "";
      } else {
	label_[a]->set(nh->nrrd->axis[a].label);
      }
      switch(nh->nrrd->axis[a].kind) {
      case nrrdKindDomain:
	kind_[a]->set("nrrdKindDomain");
	break;
      case nrrdKindScalar:
	kind_[a]->set("nrrdKindScalar");
	break;
      case nrrdKind3DSymTensor:
	kind_[a]->set("nrrdKind3DSymTensor");
	break;
      case nrrdKind3DMaskedSymTensor:
	kind_[a]->set("nrrdKind3DMaskedSymTensor");
	break;
      case nrrdKind3DTensor:
	kind_[a]->set("nrrdKind3DTensor");
	break;
      default:
	kind_[a]->set("nrrdKindUnknown");
	break;
      }
      
      switch (nh->nrrd->axis[a].center) {
      case nrrdCenterUnknown :
	center_[a]->set("Unknown");
	break;
      case nrrdCenterNode :
	center_[a]->set("Node");
	break;
      case nrrdCenterCell :
	center_[a]->set("Cell");
	break;
      }
      size_[a]->set(nh->nrrd->axis[a].size);
      spacing_[a]->set(nh->nrrd->axis[a].spacing);
      min_[a]->set(nh->nrrd->axis[a].min);
      max_[a]->set(nh->nrrd->axis[a].max);
    }
  } 

  
  if (dimension_.get() == 0) { return; }
  
  // sync with gui
  type_.reset();
  label0_.reset();
  for(int a = 0; a < dimension_.get(); a++) {
    label_[a]->reset();
    kind_[a]->reset();
    center_[a]->reset();
    size_[a]->reset();
    min_[a]->reset();
    max_[a]->reset();
    spacing_[a]->reset();
  }

  
  Nrrd *nin = nh->nrrd;
  Nrrd *nout = nrrdNew();
  
  // copy input nrrd and modify its label, min, max and spacing
  if (nrrdCopy(nout, nin)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble copying input nrrd: ") +  err);
    msgStream_ << "  input Nrrd: nin->dim=" << nin->dim << "\n";
    free(err);
    return;
  }

  int dimension = nh->nrrd->dim;
  for(int i=0; i<dimension; i++) {
    if (strlen(label_[i]->get().c_str())) {
      //AIR_FREE((void*)nout->axis[i].label);
      nout->axis[i].label = (char*)airFree(nout->axis[i].label);
      nout->axis[i].label = airStrdup(const_cast<char*>(label_[i]->get().c_str()));

      string kind = kind_[i]->get();
      if (kind == "nrrdKindDomain") {
	nout->axis[i].kind = nrrdKindDomain;
      } else if (kind == "nrrdKindScalar") {
	nout->axis[i].kind = nrrdKindScalar;
      } else if (kind == "nrrdKind3Color") {
	nout->axis[i].kind = nrrdKind3Color;
      } else if (kind == "nrrdKind3Vector") {
	nout->axis[i].kind = nrrdKind3Vector;
      } else if (kind == "nrrdKind3Normal") {
	nout->axis[i].kind = nrrdKind3Normal;
      } else if (kind == "nrrdKind3DSymTensor") {
	nout->axis[i].kind = nrrdKind3DSymTensor;
      } else if (kind == "nrrdKind3DMaskedSymTensor") {
	nout->axis[i].kind = nrrdKind3DMaskedSymTensor;
      } else if (kind == "nrrdKind3DTensor") {
	nout->axis[i].kind = nrrdKind3DTensor;
      } else {
	nout->axis[i].kind = nrrdKindUnknown;
      }


      if (AIR_EXISTS(min_[i]->get())) {
	nout->axis[i].min = min_[i]->get();
      }
      if (AIR_EXISTS(max_[i]->get())) {
	nout->axis[i].max = max_[i]->get();
      }
      if (AIR_EXISTS(spacing_[i]->get())) {
	nout->axis[i].spacing = spacing_[i]->get();
      }
    }
  }
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  // nout->axis[0].label = strdup(nin->axis[0].label);
  //nrrd->copy_sci_data(*nh.get_rep());

  oport->send(NrrdDataHandle(nrrd));
}

} // end SCITeem namespace

