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
//    File   : NrrdInfo.cc
//    Author : Martin Cole
//    Date   : Tue Feb  4 08:55:34 2003

#include <Dataflow/Network/Module.h>
#include <Dataflow/share/share.h>

#include <Teem/Core/Datatypes/NrrdData.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCITeem {
using namespace SCIRun;

using std::endl;
using std::pair;

class NrrdInfo : public Module {
public:
  NrrdInfo(GuiContext* ctx);
  virtual ~NrrdInfo();

  void clear_vals();
  void update_input_attributes(NrrdDataHandle);
  virtual void execute();

private:
  int                 generation_;
};

DECLARE_MAKER(NrrdInfo)

NrrdInfo::NrrdInfo(GuiContext* ctx)
  : Module("NrrdInfo", ctx, Source, "NrrdData", "Teem"),
    generation_(-1)
{
}

NrrdInfo::~NrrdInfo(){
}

void
NrrdInfo::clear_vals() 
{
  gui->execute(id + " delete_tabs");
#if 0
  gui->execute(string("set ") + id + "-type \"---\"");
  gui->execute(string("set ") + id + "-dimension 0");
  gui->execute(string("set ") + id + "-label0 \"---\"");

  for (int i = 1; i < nh->nrrd->dim; i++) {
    ostringstream str;
    
    str << "set " << id.c_str() << "-size" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id.c_str() << "-center" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id.c_str() << "-label" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id.c_str() << "-spacing" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id.c_str() << "-min" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id.c_str() << "-max" << i 
	<< " \"---\"";    
    gui->execute(str.str());
  }
#endif

}


void
NrrdInfo::update_input_attributes(NrrdDataHandle nh) 
{
  switch (nh->nrrd->type) {
  case nrrdTypeChar :  
    gui->execute(string("set ") + id + "-type \"char\"");
    break;
  case nrrdTypeUChar : 
    gui->execute(string("set ") + id + "-type \"unsigned char\"");
    break;
  case nrrdTypeShort : 
    gui->execute(string("set ") + id + "-type \"short\"");
    break;
  case nrrdTypeUShort :
    gui->execute(string("set ") + id + "-type \"unsigned short\"");
    break;
  case nrrdTypeInt : 
    gui->execute(string("set ") + id + "-type \"int\"");
    break;
  case nrrdTypeUInt :  
    gui->execute(string("set ") + id + "-type \"unsigned int\"");
    break;
  case nrrdTypeLLong : 
    gui->execute(string("set ") + id + "-type \"long long\"");
    break;
  case nrrdTypeULLong :
    gui->execute(string("set ") + id + "-type \"unsigned long long\"");
    break;
  case nrrdTypeFloat :
    gui->execute(string("set ") + id + "-type \"float\"");
    break;
  case nrrdTypeDouble :
    gui->execute(string("set ") + id + "-type \"double\"");
    break;
  }

  gui->execute(string("set ") + id + "-dimension " + to_string(nh->nrrd->dim));

  // Tuple Axis
  gui->execute(string("set ") + id + "-label0 {" + 
	       string(nh->nrrd->axis[0].label) + "}");

  gui->execute(string("set ") + id + "-size0 " + 
	       string(nh->nrrd->axis[0].size));

  gui->execute(id + " fill_tuple_tab");

  for (int i = 1; i < nh->nrrd->dim; i++) {
    ostringstream sz, cntr, lab, spac, min, max;
    
    sz << "set " << id.c_str() << "-size" << i 
	<< " " << nh->nrrd->axis[i].size;
    
    gui->execute(sz.str());

    cntr << "set " << id.c_str() << "-center" << i << " ";

    switch (nh->nrrd->axis[i].center) {
    case nrrdCenterUnknown :
      cntr << "Unknown";
      break;
    case nrrdCenterNode :
      cntr << "Node";
      break;
    case nrrdCenterCell :
      cntr << "Cell";
      break;
    }

    gui->execute(cntr.str());

    lab << "set " << id.c_str() << "-label" << i 
	<< " {" << nh->nrrd->axis[i].label << "}";
    gui->execute(lab.str());

    spac << "set " << id.c_str() << "-spacing" << i 
	 << " " << nh->nrrd->axis[i].spacing;
    gui->execute(spac.str());

    min << "set " << id.c_str() << "-min" << i 
	<< " " << nh->nrrd->axis[i].min;
    gui->execute(min.str());

    max << "set " << id.c_str() << "-max" << i 
	<< " " << nh->nrrd->axis[i].max;
    gui->execute(max.str());
  }

  gui->execute(id + " add_tabs");
}


void
NrrdInfo::execute()
{
  NrrdIPort *iport = (NrrdIPort*)get_iport("Query Nrrd"); 
  if (!iport) 
  {
    error("Unable to initialize iport 'Query Nrrd'.");
    return;
  }
  
  // The input port (with data) is required.
  NrrdDataHandle nh;
  if (!iport->get(nh) || !nh.get_rep())
  {
    clear_vals();
    return;
  }

  if (generation_ != nh.get_rep()->generation) 
  {
    generation_ = nh.get_rep()->generation;
    clear_vals();
    update_input_attributes(nh);
  }
}

} // end SCITeem namespace
