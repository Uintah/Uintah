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
#include <Core/Malloc/Allocator.h>
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
  gui->execute(string("set ") + id + "-type \"---\"");
  gui->execute(string("set ") + id + "-dimension \"---\"");
  gui->execute(string("set ") + id + "-size1 \"---\"");
  gui->execute(string("set ") + id + "-size2 \"---\"");
  gui->execute(string("set ") + id + "-size3 \"---\"");
  gui->execute(string("set ") + id + "-center1 \"---\"");	
  gui->execute(string("set ") + id + "-center2 \"---\"");	
  gui->execute(string("set ") + id + "-center3 \"---\"");	
  gui->execute(string("set ") + id + "-label0 \"---\"");
  gui->execute(string("set ") + id + "-label1 \"---\"");	
  gui->execute(string("set ") + id + "-label2 \"---\"");	
  gui->execute(string("set ") + id + "-label3 \"---\"");	
  gui->execute(string("set ") + id + "-spacing1 \"---\"");	
  gui->execute(string("set ") + id + "-spacing2 \"---\"");	
  gui->execute(string("set ") + id + "-spacing3 \"---\"");
  gui->execute(string("set ") + id + "-min1 \"---\"");	
  gui->execute(string("set ") + id + "-min2 \"---\"");	
  gui->execute(string("set ") + id + "-min3 \"---\"");
  gui->execute(string("set ") + id + "-max1 \"---\"");	
  gui->execute(string("set ") + id + "-max2 \"---\"");	
  gui->execute(string("set ") + id + "-max3 \"---\"");
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
  gui->execute(string("set ") + id + "-label0 " + 
	       string(nh->nrrd->axis[0].label));

  gui->execute(id + " fill_tuple_tab");
  //Axis 1
  if (nh->nrrd->dim > 1) {
    gui->execute(string("set ") + id + "-size1 " + 
		 to_string(nh->nrrd->axis[1].size));

    switch (nh->nrrd->axis[1].center) {
    case nrrdCenterUnknown :
      gui->execute(string("set ") + id + "-center1 Unknown");
      break;
    case nrrdCenterNode :
      gui->execute(string("set ") + id + "-center1 Node");
      break;
    case nrrdCenterCell :
      gui->execute(string("set ") + id + "-center1 Cell");
      break;
    }
    gui->execute(string("set ") + id + "-label1 " + 
		 string(nh->nrrd->axis[1].label));
    gui->execute(string("set ") + id + "-spacing1 " +
		 to_string(nh->nrrd->axis[1].spacing));	
    gui->execute(string("set ") + id + "-min1 " +
		 to_string(nh->nrrd->axis[1].min));	
    gui->execute(string("set ") + id + "-max1 " +
		 to_string(nh->nrrd->axis[1].max));
  }
  //Axis 2
  if (nh->nrrd->dim > 2) {
    gui->execute(string("set ") + id + "-size2 " + 
		 to_string(nh->nrrd->axis[2].size));

    switch (nh->nrrd->axis[2].center) {
    case nrrdCenterUnknown :
      gui->execute(string("set ") + id + "-center2 Unknown");
      break;
    case nrrdCenterNode :
      gui->execute(string("set ") + id + "-center2 Node");
      break;
    case nrrdCenterCell :
      gui->execute(string("set ") + id + "-center2 Cell");
      break;
    }
    gui->execute(string("set ") + id + "-label2 " + 
		 string(nh->nrrd->axis[2].label));
    gui->execute(string("set ") + id + "-spacing2 " +
		 to_string(nh->nrrd->axis[2].spacing));	
    gui->execute(string("set ") + id + "-min2 " +
		 to_string(nh->nrrd->axis[2].min));	
    gui->execute(string("set ") + id + "-max2 " +
		 to_string(nh->nrrd->axis[2].max));
  }
  //Axis 3
  if (nh->nrrd->dim > 3) {
    gui->execute(string("set ") + id + "-size3 " + 
		 to_string(nh->nrrd->axis[3].size));

    switch (nh->nrrd->axis[3].center) {
    case nrrdCenterUnknown :
      gui->execute(string("set ") + id + "-center3 Unknown");
      break;
    case nrrdCenterNode :
      gui->execute(string("set ") + id + "-center3 Node");
      break;
    case nrrdCenterCell :
      gui->execute(string("set ") + id + "-center3 Cell");
      break;
    }
    gui->execute(string("set ") + id + "-label3 " + 
		 string(nh->nrrd->axis[3].label));
    gui->execute(string("set ") + id + "-spacing3 " +
		 to_string(nh->nrrd->axis[3].spacing));	
    gui->execute(string("set ") + id + "-min3 " +
		 to_string(nh->nrrd->axis[3].min));	
    gui->execute(string("set ") + id + "-max3 " +
		 to_string(nh->nrrd->axis[3].max));	
  }

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


