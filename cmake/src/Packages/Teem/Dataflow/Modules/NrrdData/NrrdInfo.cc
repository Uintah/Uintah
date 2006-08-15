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

#include <Dataflow/Network/Ports/NrrdPort.h>

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
  void update_axis_var(const char *name, int axis, const string &val,
                       const char *pname);

  virtual void execute();

private:
  int                 generation_;
  GuiString           gui_name_;
  GuiString           gui_type_;
  GuiString           gui_dimension_;
  GuiString           gui_origin_;
};

DECLARE_MAKER(NrrdInfo)

NrrdInfo::NrrdInfo(GuiContext* ctx)
  : Module("NrrdInfo", ctx, Source, "NrrdData", "Teem"),
    generation_(-1),
    gui_name_(get_ctx()->subVar("name"), "---"),
    gui_type_(get_ctx()->subVar("type"), "---"),
    gui_dimension_(get_ctx()->subVar("dimension"), "0"),
    gui_origin_(get_ctx()->subVar("origin"), "0")
{
}


NrrdInfo::~NrrdInfo()
{
}


void
NrrdInfo::clear_vals() 
{
  gui_name_.set("---");
  gui_type_.set("---");

  get_gui()->execute(get_id() + " delete_tabs");
}


void
NrrdInfo::update_axis_var(const char *name, int axis, const string &val,
                          const char *pname)
{
  ostringstream ostr;
  ostr << "set " << get_id() << "-" << name << axis << " {" << val << "}";
  get_gui()->execute(ostr.str());
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    remark("Axis " + to_string(axis) + " " + pname + ": " + val);
  }
}


void
NrrdInfo::update_input_attributes(NrrdDataHandle nh) 
{
  const bool regressing = sci_getenv_p("SCI_REGRESSION_TESTING");

  string name;
  if (!nh->get_property( "Name", name)) { 
    name = "Unknown";
  }
  gui_name_.set(name);
  if (regressing) { remark("Name: " + name); }

  string nrrdtype, stmp;
  get_nrrd_compile_type(nh->nrrd_->type, nrrdtype, stmp);
  gui_type_.set(nrrdtype);
  if (regressing) { remark("Data Type: " + nrrdtype); }

  gui_dimension_.set(to_string(nh->nrrd_->dim));
  if (regressing) { remark("Dimension: " + to_string(nh->nrrd_->dim)); }

  // TODO: Set Origin here.

  // Tuple Axis
  for (unsigned int i = 0; i < nh->nrrd_->dim; i++)
  {
    string labelstr;
    if (nh->nrrd_->axis[i].label == NULL ||
        string(nh->nrrd_->axis[i].label).length() == 0)
    {
      labelstr = "---";
    }
    else
    {
      labelstr = nh->nrrd_->axis[i].label;
    }
    update_axis_var("label", i, labelstr, "Label");

    string kindstr;
    switch(nh->nrrd_->axis[i].kind) {
    case nrrdKindDomain:
      kindstr = "nrrdKindDomain";
      break;
    case nrrdKindScalar:
      kindstr = "nrrdKindScalar";
      break;
    case nrrdKind3Color:
      kindstr = "nrrdKind3Color";
      break;
    case nrrdKind3Vector:
      kindstr = "nrrdKind3Vector";
      break;
    case nrrdKind3Normal:
      kindstr = "nrrdKind3Normal";
      break;
    case nrrdKind3DSymMatrix:
      kindstr = "nrrdKind3DSymMatrix";
      break;
    case nrrdKind3DMaskedSymMatrix:
      kindstr = "nrrdKind3DMaskedSymMatrix";
      break;
    case nrrdKind3DMatrix:
      kindstr = "nrrdKind3DMatrix";
      break;
    case nrrdKindList:
      kindstr = "nrrdKindList";
      break;
    case nrrdKindStub:
      kindstr = "nrrdKindStub";
      break;	
    default:
      nh->nrrd_->axis[i].kind = nrrdKindUnknown;
      kindstr = "nrrdKindUnknown";
      break;
    }
    update_axis_var("kind", i, kindstr, "Kind");

    update_axis_var("size", i, to_string(nh->nrrd_->axis[i].size), "Size");

    update_axis_var("min", i, to_string(nh->nrrd_->axis[i].min), "Min");
    update_axis_var("max", i, to_string(nh->nrrd_->axis[i].max), "Max");

    string locstr;
    switch (nh->nrrd_->axis[i].center) {
    case nrrdCenterUnknown :
      locstr = "Unknown";
      break;
    case nrrdCenterNode :
      locstr = "Node";
      break;
    case nrrdCenterCell :
      locstr = "Cell";
      break;
    }
    update_axis_var("center", i, locstr, "Center");

    update_axis_var("spacing", i, to_string(nh->nrrd_->axis[i].spacing), "Spacing");
    
    string spacedir = "[ ";
    for (int p=0; p<nh->nrrd_->spaceDim; p++)
    {
      std::ostringstream oss;
      oss << nh->nrrd_->axis[i].spaceDirection[p];
      spacedir += oss.str();
      if (p == nh->nrrd_->dim-1) spacedir += " ]"; else spacedir += ", ";
    }
    update_axis_var("spaceDir", i, spacedir, "Spacing Direction");
  }

  get_gui()->execute(get_id() + " add_tabs");
}


void
NrrdInfo::execute()
{
  NrrdDataHandle nh;
  if (!get_input_handle("Query Nrrd", nh))
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
