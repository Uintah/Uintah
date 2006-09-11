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

#include <Core/Algorithms/Converter/ConverterAlgo.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

namespace ModelCreation {

using namespace SCIRun;

class FieldInfo : public Module {
private:
  GuiString gui_fldname_;
  GuiString gui_generation_;
  GuiString gui_typename_;
  GuiString gui_datamin_;
  GuiString gui_datamax_;
  GuiString gui_numnodes_;
  GuiString gui_numelems_;
  GuiString gui_numdata_;
  GuiString gui_dataat_;
  GuiString gui_cx_;
  GuiString gui_cy_;
  GuiString gui_cz_;
  GuiString gui_sizex_;
  GuiString gui_sizey_;
  GuiString gui_sizez_;

  double min_;
  double max_;
  Point  center_;
  Vector size_;
  int    numelems_;
  int    numnodes_;
  int    numdata_;
  
  void clear_vals();
  void update_input_attributes(FieldHandle);

public:
  FieldInfo(GuiContext* ctx);
  virtual void execute();
};


DECLARE_MAKER(FieldInfo)

FieldInfo::FieldInfo(GuiContext* ctx)
  : Module("FieldInfo", ctx, Sink, "DataInfo", "ModelCreation"),
    gui_fldname_(get_ctx()->subVar("fldname", false),"---"),
    gui_generation_(get_ctx()->subVar("generation", false),"---"),
    gui_typename_(get_ctx()->subVar("typename", false),"---"),
    gui_datamin_(get_ctx()->subVar("datamin", false),"---"),
    gui_datamax_(get_ctx()->subVar("datamax", false),"---"),
    gui_numnodes_(get_ctx()->subVar("numnodes", false),"---"),
    gui_numelems_(get_ctx()->subVar("numelems", false),"---"),
    gui_numdata_(get_ctx()->subVar("numdata", false),"---"),
    gui_dataat_(get_ctx()->subVar("dataat", false),"---"),
    gui_cx_(get_ctx()->subVar("cx", false),"---"),
    gui_cy_(get_ctx()->subVar("cy", false),"---"),
    gui_cz_(get_ctx()->subVar("cz", false),"---"),
    gui_sizex_(get_ctx()->subVar("sizex", false),"---"),
    gui_sizey_(get_ctx()->subVar("sizey", false),"---"),
    gui_sizez_(get_ctx()->subVar("sizez", false),"---"),
    min_(0.0),
    max_(0.0),
    numelems_(0),
    numnodes_(0),
    numdata_(0)
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
  gui_numdata_.set("---");
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
  // Name
  std::string fldname;
  if (f->get_property("name",fldname))
  {
    gui_fldname_.set(fldname);
  }
  else
  {
    gui_fldname_.set("--- Name Not Assigned ---");
  }

  // Generation
  gui_generation_.set(to_string(f->generation));

  // Typename
  const std::string &tname = f->get_type_description()->get_name();
  gui_typename_.set(tname);

  // Basis
  static char *at_table[4] = { "Nodes", "Edges", "Faces", "Cells" };
  switch(f->basis_order())
  {
  case 1:
    gui_dataat_.set("Nodes (linear basis)");
    break;
  case 0:
    gui_dataat_.set(at_table[f->mesh()->dimensionality()] +
                    string(" (constant basis)"));
    break;
  case -1:
    gui_dataat_.set("None");
    break;
  }

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
    
    size_ = size;
    center_ = center;
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

    size_ = Vector(0.0,0.0,0.0);
    center_ = Point(0.0,0.0,0.0);
  }

  ScalarFieldInterfaceHandle sdi = f->query_scalar_interface(this);
  if (sdi.get_rep())
  {
    std::pair<double, double> minmax;
    sdi->compute_min_max(minmax.first,minmax.second);
    gui_datamin_.set(to_string(minmax.first));
    gui_datamax_.set(to_string(minmax.second));
    
    min_ = minmax.first;
    max_ = minmax.second;
  }
  else
  {
    gui_datamin_.set("--- N/A ---");
    gui_datamax_.set("--- N/A ---");
    
    min_ = 0.0;
    max_ = 0.0;
  }

  SCIRunAlgo::FieldsAlgo algo(this);
  if(!(algo.GetFieldInfo(f,numnodes_,numelems_))) return;

  switch(f->basis_order())
  {
  case 1:
    numdata_ = numnodes_;
    break;
  case 0:
    numdata_ = numelems_;
    break;
  default:
    numdata_ = 0;
    break;
  }

  std::ostringstream num_nodes; num_nodes << numnodes_;
  std::ostringstream num_elems; num_elems << numelems_;
  std::ostringstream num_data;  num_data  << numdata_;
  
  gui_numnodes_.set(num_nodes.str());
  gui_numelems_.set(num_elems.str());
  gui_numdata_.set(num_data.str());
}


void
FieldInfo::execute()
{  
  FieldHandle fh;
  
  if (!(get_input_handle("Input Field",fh,true)))
  {
    clear_vals();
    return;
  }

  if (inputs_changed_ || !oport_cached("NumNodes") || 
      !oport_cached("NumElements") || !oport_cached("NumData") ||
      !oport_cached("DataMin") || !oport_cached("DataMax") ||
      !oport_cached("FieldSize") || !oport_cached("FieldCenter") )
  {
    update_input_attributes(fh);

    MatrixHandle NumNodes, NumElements, NumData, DataMin, DataMax, FieldSize, FieldCenter;
    
    SCIRunAlgo::ConverterAlgo calgo(this);

    if (!(calgo.IntToMatrix(numnodes_,NumNodes))) return;
    if (!(calgo.IntToMatrix(numelems_,NumElements))) return;
    if (!(calgo.IntToMatrix(numdata_,NumData))) return;
    if (!(calgo.DoubleToMatrix(min_,DataMin))) return;
    if (!(calgo.DoubleToMatrix(max_,DataMax))) return;
    if (!(calgo.VectorToMatrix(size_,FieldSize))) return;
    if (!(calgo.PointToMatrix(center_,FieldCenter))) return;
    
    send_output_handle("NumNodes", NumNodes);
    send_output_handle("NumElements", NumElements);
    send_output_handle("NumData", NumData);
    send_output_handle("DataMin", DataMin);
    send_output_handle("DataMax", DataMax);
    send_output_handle("FieldSize", FieldSize);
    send_output_handle("FieldCenter", FieldCenter);
  }
}

} // end SCIRun namespace


