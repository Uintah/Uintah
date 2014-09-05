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

#include <Core/Algorithms/Fields/FieldCount.h>
#include <Core/Containers/Handle.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <map>
#include <iostream>
#include <sstream>

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

  int generation_;
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
  virtual ~FieldInfo();
  virtual void execute();
};

  DECLARE_MAKER(FieldInfo)

FieldInfo::FieldInfo(GuiContext* ctx)
  : Module("FieldInfo", ctx, Sink, "DataInfo", "ModelCreation"),
    gui_fldname_(ctx->subVar("fldname", false)),
    gui_generation_(ctx->subVar("generation", false)),
    gui_typename_(ctx->subVar("typename", false)),
    gui_datamin_(ctx->subVar("datamin", false)),
    gui_datamax_(ctx->subVar("datamax", false)),
    gui_numnodes_(ctx->subVar("numnodes", false)),
    gui_numelems_(ctx->subVar("numelems", false)),
    gui_numdata_(ctx->subVar("numdata", false)),
    gui_dataat_(ctx->subVar("dataat", false)),
    gui_cx_(ctx->subVar("cx", false)),
    gui_cy_(ctx->subVar("cy", false)),
    gui_cz_(ctx->subVar("cz", false)),
    gui_sizex_(ctx->subVar("sizex", false)),
    gui_sizey_(ctx->subVar("sizey", false)),
    gui_sizez_(ctx->subVar("sizez", false)),
    generation_(-1),
    min_(0.0),
    max_(0.0),
    numelems_(0),
    numnodes_(0),
    numdata_(0)
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
  string fldname;
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
  const string &tname = f->get_type_description()->get_name();
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

  // Do this last, sometimes takes a while.
  const TypeDescription *meshtd = f->mesh()->get_type_description();
  CompileInfoHandle ci = FieldCountAlgorithm::get_compile_info(meshtd);
  Handle<FieldCountAlgorithm> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  //string num_nodes, num_elems;
  //int num_nodes, num_elems;
  const string num_nodes = algo->execute_node(f->mesh());
  const string num_elems = algo->execute_elem(f->mesh());
  string num_data;
  
  switch(f->basis_order())
  {
  case 1:
    num_data = num_nodes;
    break;
  case 0:
    num_data = num_elems;
    break;
  default:
    num_data = "0";
    break;
  }
  
  gui_numnodes_.set(num_nodes);
  gui_numelems_.set(num_elems);
  gui_numdata_.set(num_data);

  std::istringstream iss_nnodes(num_nodes);
  std::istringstream iss_nelems(num_elems);
  iss_nnodes >> numnodes_;
  iss_nelems >> numelems_;
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
}


void
FieldInfo::execute()
{  
  FieldIPort *iport = (FieldIPort*)get_iport("Input Field");

  // The input port (with data) is required.
  FieldHandle fh;
  if (!iport->get(fh) || !fh.get_rep())
  {
    clear_vals();
    generation_ = -1;
    return;
  }

  if (generation_ != fh.get_rep()->generation)
  {
    generation_ = fh.get_rep()->generation;
    update_input_attributes(fh);
  }

  MatrixOPort* oport;

  if (oport = dynamic_cast<MatrixOPort *>(get_oport("NumNodes")))
  {
    MatrixHandle nnodes = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if(nnodes.get_rep() == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }
    double* dataptr = nnodes->get_data_pointer();
    if (dataptr == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }    
    dataptr[0] = static_cast<double>(numnodes_);
    oport->send(nnodes);
  }

  if (oport = dynamic_cast<MatrixOPort *>(get_oport("NumElements")))
  {
    MatrixHandle nelems = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if(nelems.get_rep() == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }
    double* dataptr = nelems->get_data_pointer();
    if (dataptr == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }    
    dataptr[0] = static_cast<double>(numelems_);
    oport->send(nelems);
  }

  if (oport = dynamic_cast<MatrixOPort *>(get_oport("NumData")))
  {
    MatrixHandle ndata = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if(ndata.get_rep() == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }
    double* dataptr = ndata->get_data_pointer();
    if (dataptr == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }    
    dataptr[0] = static_cast<double>(numdata_);
    oport->send(ndata);
  }

  if (oport = dynamic_cast<MatrixOPort *>(get_oport("DataMin")))
  {
    MatrixHandle data = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if(data.get_rep() == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }
    double* dataptr = data->get_data_pointer();
    if (dataptr == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }    
    dataptr[0] = static_cast<double>(min_);
    oport->send(data);
  }

  if (oport = dynamic_cast<MatrixOPort *>(get_oport("DataMax")))
  {
    MatrixHandle data = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if(data.get_rep() == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }
    double* dataptr = data->get_data_pointer();
    if (dataptr == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }    
    dataptr[0] = static_cast<double>(max_);
    oport->send(data);
  }

  if (oport = dynamic_cast<MatrixOPort *>(get_oport("FieldSize")))
  {
    MatrixHandle data = dynamic_cast<Matrix *>(scinew DenseMatrix(1,3));
    if(data.get_rep() == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }
    double* dataptr = data->get_data_pointer();
    if (dataptr == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }    
    dataptr[0] = static_cast<double>(size_.x());
    dataptr[1] = static_cast<double>(size_.y());
    dataptr[2] = static_cast<double>(size_.z());
    oport->send(data);
  }

  if (oport = dynamic_cast<MatrixOPort *>(get_oport("FieldCenter")))
  {
    MatrixHandle data = dynamic_cast<Matrix *>(scinew DenseMatrix(1,3));
    if(data.get_rep() == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }
    double* dataptr = data->get_data_pointer();
    if (dataptr == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }    
    dataptr[0] = static_cast<double>(center_.x());
    dataptr[1] = static_cast<double>(center_.y());
    dataptr[2] = static_cast<double>(center_.z());
    oport->send(data);
  }


}




} // end SCIRun namespace


