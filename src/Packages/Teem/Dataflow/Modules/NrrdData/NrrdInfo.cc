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

//    File   : NrrdInfo.cc
//    Author : Martin Cole
//    Date   : Tue Feb  4 08:55:34 2003

#include <Dataflow/Network/Module.h>

#include <Dataflow/Ports/NrrdPort.h>

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
  gui->execute(string("set ") + id + "-name \"---\"");

  gui->execute(id + " delete_tabs");

#if 0
  gui->execute(string("set ") + id + "-type \"---\"");
  gui->execute(string("set ") + id + "-dimension 0");
  gui->execute(string("set ") + id + "-origin \"---\"");
  gui->execute(string("set ") + id + "-label0 \"---\"");
  gui->execute(string("set ") + id + "-kind0 \"---\"");

  for (int i = 0; i < nh->nrrd->dim; i++) {
    ostringstream str;
    
    str << "set " << id << "-size" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id << "-center" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id << "-label" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id << "-kind" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id << "-spacing" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id << "-min" << i 
	<< " \"---\"";    
    gui->execute(str.str());
    str.clear();

    str << "set " << id << "-max" << i 
	<< " \"---\"";    
    gui->execute(str.str());

    str << "set " << id << "-spaceDir" << i 
	<< " \"---\"";    
    gui->execute(str.str());
  }
#endif

}


void
NrrdInfo::update_input_attributes(NrrdDataHandle nh) 
{
  string name;
  if (nh->get_property( "Name", name)) { 
    gui->execute(string("set ") + id + "-name \"" + name+"\"");
  } else {
    gui->execute(string("set ") + id + "-name \"Unknown\"");
  }

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


  int space_axis = 0;
  for (int i = 0; i < nh->nrrd->dim; i++) {
    ostringstream sz, cntr, lab, kind, spac, min, max, spaceDir;
    
    sz << "set " << id << "-size" << i 
	<< " " << nh->nrrd->axis[i].size;
    
    gui->execute(sz.str());

    cntr << "set " << id << "-center" << i << " ";

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

    if (nh->nrrd->axis[i].label == NULL || string(nh->nrrd->axis[i].label).length() == 0) {
      lab << "set " << id << "-label" << i 
	  << " {" << "---" << "}";
      gui->execute(lab.str());
    } else {
      lab << "set " << id << "-label" << i 
	  << " {" << nh->nrrd->axis[i].label << "}";
      gui->execute(lab.str());
    }

    switch(nh->nrrd->axis[i].kind) {
    case nrrdKindDomain:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKindDomain}";
      gui->execute(kind.str());
      break;
    case nrrdKindScalar:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKindScalar}";
      gui->execute(kind.str());
      break;
    case nrrdKind3Color:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKind3Color}";
      gui->execute(kind.str());
      break;
    case nrrdKind3Vector:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKind3Vector}";
      gui->execute(kind.str());
      break;
    case nrrdKind3Normal:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKind3Normal}";
      gui->execute(kind.str());
      break;
    case nrrdKind3DSymMatrix:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKind3DSymMatrix}";
      gui->execute(kind.str());
      break;
    case nrrdKind3DMaskedSymMatrix:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKind3DMaskedSymMatrix}";
      gui->execute(kind.str());
      break;
    case nrrdKind3DMatrix:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKind3DMatrix}";
      gui->execute(kind.str());
      break;
    case nrrdKindList:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKindList}";
      gui->execute(kind.str());
      break;	
    case nrrdKindStub:
      kind << "set " << id << "-kind" << i 
           << " {nrrdKindStub}";
      gui->execute(kind.str());
      break;	
    default:
      nh->nrrd->axis[i].kind = nrrdKindUnknown;
      kind << "set " << id << "-kind" << i 
           << " {" << "nrrdKindUnknown" << "}";
      gui->execute(kind.str());
      break;
    }

    // use spaceOrigin and spaceDirection information
    // if available, else use min/max and spacing
    double spacing = 0;
    double *vec = new double[NRRD_SPACE_DIM_MAX];
    
    int result = nrrdSpacingCalculate(nh->nrrd, i, &spacing, vec);
    if (result ==  nrrdSpacingStatusDirection &&
	airExists(nh->nrrd->axis[i].spaceDirection[0]) &&
	airExists(nh->nrrd->spaceOrigin[space_axis]))
      
    {
      string space_origin_string = "\"(";
      for (int d=0; d<nh->nrrd->spaceDim; d++) {
	space_origin_string += to_string(nh->nrrd->spaceOrigin[d]);
	if (d < (nh->nrrd->spaceDim-1))
	  space_origin_string += ", ";
      }
      space_origin_string += ")\"";
      gui->execute(string("set ") + id + "-origin " + space_origin_string);
      
      
      spac << "set " << id << "-spacing" << i 
	   << " " << spacing;
      gui->execute(spac.str());
      
      min << "set " << id << "-min" << i 
	  << " \"not available\"";
      gui->execute(min.str());
      
      max << "set " << id << "-max" << i 
	  << " \"not available\"";
      gui->execute(max.str());
      
      string space_dir_string = "\"(";
      for (int d=0; d<nh->nrrd->spaceDim; d++) {
	space_dir_string += to_string(nh->nrrd->axis[i].spaceDirection[d]);
	if (d < (nh->nrrd->spaceDim-1))
	  space_dir_string += ", ";
      }
      space_dir_string += ")\"";

      spaceDir << "set " << id << "-spaceDir" << i 
	  << " " << space_dir_string;
      gui->execute(spaceDir.str());

      space_axis++;
    } else
    {
      gui->execute(string("set ") + id + "-origin " + "\"not available\"");

      spac << "set " << id << "-spacing" << i 
	   << " " << nh->nrrd->axis[i].spacing;
      gui->execute(spac.str());
      
      min << "set " << id << "-min" << i 
	  << " " << nh->nrrd->axis[i].min;
      gui->execute(min.str());
      
      max << "set " << id << "-max" << i 
	  << " " << nh->nrrd->axis[i].max;
      gui->execute(max.str());

      spaceDir << "set " << id << "-spaceDir" << i 
	       << " \"not available\"";
      gui->execute(spaceDir.str());

    }
    delete vec;
  }

  gui->execute(id + " add_tabs");
}


void
NrrdInfo::execute()
{
  NrrdIPort *iport = (NrrdIPort*)get_iport("Query Nrrd"); 
  
  // The input port (with data) is required.
  NrrdDataHandle nh;
  if (!iport->get(nh) || !nh.get_rep())
  {
    clear_vals();
    generation_ = -1;
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
