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
  vector<GuiString*>  label_;
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
    for(int a = max_vectors_+1; a <= dimension_.get(); a++) {
      ostringstream lab, cntr, sz, min, max, spac;
      lab << "label" << a;
      label_.push_back(new GuiString(ctx->subVar(lab.str())));
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
    return;
  }

  dimension_.reset();
  
  if (generation_ != nh.get_rep()->generation) 
  {
    load_gui();

    // if the dimension, and sizes
    // don't clear
    bool do_clear = false;
    if(dimension_.get() == nh->nrrd->dim) {
      for(int a = 1; a < dimension_.get(); a++) {
	if(size_[a-1]->get() != nh->nrrd->axis[a].size) {
	  do_clear = true;
	  break;
	}
      }
    } else {
      do_clear = true;
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

      
      // Tuple Axis information 
      label0_.set(nh->nrrd->axis[0].label);

      gui->execute(id.c_str() + string(" init_axes"));

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
      for(int a = 1; a < dimension_.get(); a++) {
	label_[a-1]->set(nh->nrrd->axis[a].label);
	switch (nh->nrrd->axis[a].center) {
	case nrrdCenterUnknown :
	  center_[a-1]->set("Unknown");
	  break;
	case nrrdCenterNode :
	  center_[a-1]->set("Node");
	  break;
	case nrrdCenterCell :
	  center_[a-1]->set("Cell");
	  break;
	}
	size_[a-1]->set(nh->nrrd->axis[a].size);
	spacing_[a-1]->set(nh->nrrd->axis[a].spacing);
	min_[a-1]->set(nh->nrrd->axis[a].min);
	max_[a-1]->set(nh->nrrd->axis[a].max);
      }
      
    }
  }
  
  if (dimension_.get() == 0) { return; }

  // sync with gui
  type_.reset();
  label0_.reset();
  for(int a = 0; a < dimension_.get(); a++) {
    label_[a]->reset();
    center_[a]->reset();
    size_[a]->reset();
    min_[a]->reset();
    max_[a]->reset();
    spacing_[a]->reset();
  }


  generation_ = nh->generation;
  
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
  for(int i=1; i<dimension; i++) {
    // the gui vectors are 1 off because the axes
    // start with 0, with 0 being the tuple axis.
    // The vectors only contain information for non-tuple
    // axes.
    int index = i-1;

    if (strlen(label_[index]->get().c_str())) {
      //AIR_FREE((void*)nout->axis[i].label);
      nout->axis[i].label = (char*)airFree(nout->axis[i].label);
      nout->axis[i].label = airStrdup(const_cast<char*>(label_[index]->get().c_str()));

      if (AIR_EXISTS(min_[index]->get())) {
	nout->axis[i].min = min_[index]->get();
      }
      if (AIR_EXISTS(max_[index]->get())) {
	nout->axis[i].max = max_[index]->get();
      }
      if (AIR_EXISTS(spacing_[index]->get())) {
	nout->axis[i].spacing = spacing_[index]->get();
      }
    }
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  nout->axis[0].label = strdup(nin->axis[0].label);
  nrrd->copy_sci_data(*nh.get_rep());

  oport->send(NrrdDataHandle(nrrd));
}

} // end SCITeem namespace

