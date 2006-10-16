/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


//    File   : FieldInfo.cc
//    Author : McKay Davis
//    Date   : July 2002

#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Containers/Handle.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <map>
#include <iostream>
#include <sstream>

namespace SCIRun {

using std::endl;
using std::pair;

class FieldInfo : public Module {
private:
  GuiString gui_fldname_;
  GuiString gui_generation_;
  GuiString gui_typename_;
  GuiString gui_datamin_;
  GuiString gui_datamax_;
  GuiString gui_numnodes_;
  GuiString gui_numelems_;
  GuiString gui_dataat_;
  GuiString gui_cx_;
  GuiString gui_cy_;
  GuiString gui_cz_;
  GuiString gui_sizex_;
  GuiString gui_sizey_;
  GuiString gui_sizez_;

  void clear_vals();
  void update_input_attributes(FieldHandle);

public:
  FieldInfo(GuiContext* ctx);
  virtual ~FieldInfo();
  virtual void execute();
};

  DECLARE_MAKER(FieldInfo)

FieldInfo::FieldInfo(GuiContext* ctx)
  : Module("FieldInfo", ctx, Sink, "FieldsOther", "SCIRun"),
    gui_fldname_(get_ctx()->subVar("fldname", false), "---"),
    gui_generation_(get_ctx()->subVar("generation", false), "---"),
    gui_typename_(get_ctx()->subVar("typename", false), "---"),
    gui_datamin_(get_ctx()->subVar("datamin", false), "---"),
    gui_datamax_(get_ctx()->subVar("datamax", false), "---"),
    gui_numnodes_(get_ctx()->subVar("numnodes", false), "---"),
    gui_numelems_(get_ctx()->subVar("numelems", false), "---"),
    gui_dataat_(get_ctx()->subVar("dataat", false), "---"),
    gui_cx_(get_ctx()->subVar("cx", false), "---"),
    gui_cy_(get_ctx()->subVar("cy", false), "---"),
    gui_cz_(get_ctx()->subVar("cz", false), "---"),
    gui_sizex_(get_ctx()->subVar("sizex", false), "---"),
    gui_sizey_(get_ctx()->subVar("sizey", false), "---"),
    gui_sizez_(get_ctx()->subVar("sizez", false), "---")
{
}


FieldInfo::~FieldInfo()
{
}



void
FieldInfo::clear_vals()
{
  gui_fldname_.set("---");
  gui_generation_.set("---");
  gui_typename_.set("---");
  gui_datamin_.set("---");
  gui_datamax_.set("---");
  gui_numnodes_.set("---");
  gui_numelems_.set("---");
  gui_dataat_.set("---");
  gui_cx_.set("---");
  gui_cy_.set("---");
  gui_cz_.set("---");
  gui_sizex_.set("---");
  gui_sizey_.set("---");
  gui_sizez_.set("---");
}


void
FieldInfo::update_input_attributes(FieldHandle f)
{
  const bool regressing = sci_getenv_p("SCI_REGRESSION_TESTING");
  
  // Name
  string fldname;
  if (!f->get_property("name",fldname))
  {
    fldname = "--- Name Not Assigned ---";
  }
  gui_fldname_.set(fldname);
  if (regressing) { remark("Name: " + fldname); }

  // Generation
  const string gen = to_string(f->generation);
  gui_generation_.set(gen);
  if (regressing) { remark("Generation: " + gen); }

  // Typename
  const string &tname = f->get_type_description()->get_name();
  gui_typename_.set(tname);
  if (regressing) { remark("Type: " + tname); }

  // Basis
  static char *at_table[4] = { "Nodes", "Edges", "Faces", "Cells" };
  string dataat;
  switch(f->basis_order())
  {
  case 1:
    dataat = "Nodes (linear basis)";
    break;
  case 0:
    dataat = at_table[f->mesh()->dimensionality()] +
      string(" (constant basis)");
    break;
  case -1:
    dataat = "None";
    break;
  default:
    dataat = "High Order Basis";
    break;
  }
  gui_dataat_.set(dataat);
  if (regressing) { remark("Data At: " + dataat); }

  Point center;
  Vector size;

  const BBox bbox = f->mesh()->get_bounding_box();
  if (bbox.valid())
  {
    size = bbox.diagonal();
    center = bbox.center();
    gui_cx_.set(to_string(center.x()));
    gui_cy_.set(to_string(center.y()));
    gui_cz_.set(to_string(center.z()));
    gui_sizex_.set(to_string(size.x()));
    gui_sizey_.set(to_string(size.y()));
    gui_sizez_.set(to_string(size.z()));
    if (regressing)
    {
      remark("Center: "
             + to_string(center.x()) + " "
             + to_string(center.y()) + " "
             + to_string(center.z()));
      remark("Size: "
             + to_string(size.x()) + " "
             + to_string(size.y()) + " "
             + to_string(size.z()));
    }
  }
  else
  {
    warning("Input Field is empty.");
    gui_cx_.set("--- N/A ---");
    gui_cy_.set("--- N/A ---");
    gui_cz_.set("--- N/A ---");
    gui_sizex_.set("--- N/A ---");
    gui_sizey_.set("--- N/A ---");
    gui_sizez_.set("--- N/A ---");
  }

  ScalarFieldInterfaceHandle sdi = f->query_scalar_interface(this);
  if (sdi.get_rep())
  {
    pair<double, double> minmax;
    sdi->compute_min_max(minmax.first,minmax.second);
    gui_datamin_.set(to_string(minmax.first));
    gui_datamax_.set(to_string(minmax.second));
    if (regressing)
    {
      remark("Data Min: " + to_string(minmax.first));
      remark("Data Max: " + to_string(minmax.second));
    }
  }
  else
  {
    gui_datamin_.set("--- N/A ---");
    gui_datamax_.set("--- N/A ---");
  }

  // Do this last, sometimes takes a while.

  int nnodes, nelems;

  SCIRunAlgo::FieldsAlgo algo(this);
  algo.GetFieldInfo(f,nnodes,nelems);
  
  std::ostringstream num_nodes; num_nodes << nnodes;
  std::ostringstream num_elems; num_elems << nelems;

  gui_numnodes_.set(num_nodes.str());
  gui_numelems_.set(num_elems.str());
  if (regressing)
  {
    remark("Num Nodes: " + num_nodes.str());
    remark("Num Elems: " + num_elems.str());
  }
}


void
FieldInfo::execute()
{
  FieldHandle field_input_handle;
  if( !get_input_handle( "Input Field", field_input_handle, true ) ) {
    clear_vals();
    return;
  }

  // If no data or a changed recalcute.
  if( inputs_changed_ )
    update_input_attributes(field_input_handle);
}




} // end SCIRun namespace


