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
 *  SampleStructHex.cc:  Make an ImageField that fits the source field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

class SampleStructHex : public Module
{
public:
  typedef StructHexVolMesh<HexTrilinearLgn<Point> > SHVMesh;

  SampleStructHex(GuiContext* ctx);
  virtual ~SampleStructHex();

  virtual void execute();

private:
  GuiInt size_x_;
  GuiInt size_y_;
  GuiInt size_z_;
  GuiDouble padpercent_;
  GuiString data_at_;

  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };
};


DECLARE_MAKER(SampleStructHex)

SampleStructHex::SampleStructHex(GuiContext* ctx)
  : Module("SampleStructHex", ctx, Filter, "FieldsCreate", "SCIRun"),
    size_x_(get_ctx()->subVar("sizex"), 16),
    size_y_(get_ctx()->subVar("sizey"), 16),
    size_z_(get_ctx()->subVar("sizez"), 16),
    padpercent_(get_ctx()->subVar("padpercent"), 0.0),
    data_at_(get_ctx()->subVar("data-at"), "Nodes")
{
}


#if 0
static Point
user_cylinder_transform(const Point &p)
{
  double phi, r, uu, vv;

  const double a = p.x();
  const double b = p.y();
  const double c = p.z();
  
  if (a > -b)
  {
    if (a > b)
    {
      r = a;
      phi = (M_PI/4.0) * b/a;
    }
    else
    {
      r = b;
      phi = (M_PI/4.0) * (2 - a/b);
    }
  }
  else
  {
    if (a < b)
    {
      r = -a;
      phi = (M_PI/4.0) * (4 + (b/a));
    }
    else
    {
      r = -b;
      if (b != 0)
      {
	phi = (M_PI/4.0) * (6 - a/b);
      }
      else
      {
	phi = 0;
      }
    }
  }

  uu = r * cos(phi);
  vv = r * sin(phi);
  
  return Point(uu, vv, c);
}



static Point
user_sphere_transform(const Point &p)
{
  double phi, r, u, v;

  const double a = p.x();
  const double b = p.y();
  const double c = p.z();
  
  if (a > -b)
  {
    if (a > b)
    {
      r = a;
      phi = (M_PI/4.0) * b/a;
    }
    else
    {
      r = b;
      phi = (M_PI/4.0) * (2 - a/b);
    }
  }
  else
  {
    if (a < b)
    {
      r = -a;
      phi = (M_PI/4.0) * (4 + (b/a));
    }
    else
    {
      r = -b;
      if (b != 0)
      {
	phi = (M_PI/4.0) * (6 - a/b);
      }
      else
      {
	phi = 0;
      }
    }
  }

  u = r * cos(phi);
  v = r * sin(phi);
  
  return Point(u, v, c);
}
#endif


SampleStructHex::~SampleStructHex()
{
}


void
SampleStructHex::execute()
{
  FieldHandle ifieldhandle;
  Point minb, maxb;
  DataTypeEnum datatype;
  if (!get_input_handle("Input Field", ifieldhandle, false))
  {
    datatype = SCALAR;
    minb = Point(-1.0, -1.0, -1.0);
    maxb = Point(1.0, 1.0, 1.0);
  }
  else
  {
    datatype = SCALAR;
    if (ifieldhandle->query_vector_interface(this).get_rep())
    {
      datatype = VECTOR;
    }
    else if (ifieldhandle->query_tensor_interface(this).get_rep())
    {
      datatype = TENSOR;
    }
    BBox bbox = ifieldhandle->mesh()->get_bounding_box();
    minb = bbox.min();
    maxb = bbox.max();
  }

  Vector diag((maxb.asVector() - minb.asVector()) * (padpercent_.get()/100.0));
  minb -= diag;
  maxb += diag;

  // Create blank mesh.
  unsigned int sizex = Max(2, size_x_.get());
  unsigned int sizey = Max(2, size_y_.get());
  unsigned int sizez = Max(2, size_z_.get());
  SHVMesh::handle_type mesh = scinew SHVMesh(sizex, sizey, sizez);

  Transform trans;
  trans.pre_scale(Vector(1.0 / (sizex-1.0),
			 1.0 / (sizey-1.0),
			 1.0 / (sizez-1.0)));
  trans.pre_scale(maxb - minb);
  trans.pre_translate(minb.asVector());

  SHVMesh::Node::iterator mitr, mitr_end;
  mesh->begin(mitr);
  mesh->end(mitr_end);
  while (mitr != mitr_end)
  {
    const Point p0((*mitr).i_, (*mitr).j_, (*mitr).k_);
    const Point p = trans.project(p0);
    mesh->set_point(p, *mitr);
    ++mitr;
  }
  mesh->set_transform(trans);

  int basis_order;
  if (data_at_.get() == "Nodes") basis_order = 1;
  else if (data_at_.get() == "Cells") basis_order = 0;
  else if (data_at_.get() == "None") basis_order = -1;
  else {
    error("Unsupported data_at location " + data_at_.get() + ".");
    return;
  }

  // Create Image Field.
  FieldHandle ofh;
  if (datatype == SCALAR)
  {
    typedef NoDataBasis<double>                 NBasis;
    typedef ConstantBasis<double>               CBasis;
    typedef HexTrilinearLgn<double>             LBasis;
    
    if (basis_order == -1) {
      typedef GenericField<SHVMesh, NBasis, FData3d<double,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      ofh = lvf;
    } else if (basis_order == 0) {
      typedef GenericField<SHVMesh, CBasis, FData3d<double,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      SHVField::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = 0.0;
	++itr;
      }
      ofh = lvf;
    } else {
      typedef GenericField<SHVMesh, LBasis, FData3d<double,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      SHVField::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = 0.0;
	++itr;
      }
      ofh = lvf;
    }
  } 
  else if (datatype == VECTOR)
  {
    typedef NoDataBasis<Vector>                 NBasis;
    typedef ConstantBasis<Vector>               CBasis;
    typedef HexTrilinearLgn<Vector>             LBasis;
    
    if (basis_order == -1) {
      typedef GenericField<SHVMesh, NBasis, FData3d<Vector,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      ofh = lvf;
    } else if (basis_order == 0) {
      typedef GenericField<SHVMesh, CBasis, FData3d<Vector,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      SHVField::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = Vector(0.0, 0.0, 0.0);
	++itr;
      }
      ofh = lvf;
    } else {
      typedef GenericField<SHVMesh, LBasis, FData3d<Vector,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      SHVField::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = Vector(0.0, 0.0, 0.0);
	++itr;
      }
      ofh = lvf;
    }
  }				    
  else // if (datatype == TENSOR)	    
  {	
    typedef NoDataBasis<Tensor>                 NBasis;
    typedef ConstantBasis<Tensor>               CBasis;
    typedef HexTrilinearLgn<Tensor>             LBasis;
    
    if (basis_order == -1) {
      typedef GenericField<SHVMesh, NBasis, FData3d<Tensor,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      ofh = lvf;
    } else if (basis_order == 0) {
      typedef GenericField<SHVMesh, CBasis, FData3d<Tensor,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      SHVField::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = Tensor(0.0);
	++itr;
      }
      ofh = lvf;
    } else {
      typedef GenericField<SHVMesh, LBasis, FData3d<Tensor,SHVMesh> > SHVField;
      SHVField *lvf = scinew SHVField(mesh);
      SHVField::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = Tensor(0.0);
	++itr;
      }
      ofh = lvf;
    }
  }				    

  send_output_handle("Output Sample Field", ofh);
}


} // End namespace SCIRun

