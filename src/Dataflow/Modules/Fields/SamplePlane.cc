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
 *  SamplePlane.cc:  Make an ImageField that fits the source field.
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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class SamplePlane : public Module
{
public:
  SamplePlane(GuiContext* ctx);
  virtual ~SamplePlane();

  virtual void execute();

private:

  GuiInt size_x_;
  GuiInt size_y_;
  GuiInt auto_size_;
  GuiInt axis_;
  GuiDouble padpercent_;
  GuiDouble position_;
  GuiString data_at_;
  GuiString update_type_;
  GuiPoint custom_origin_;
  GuiVector custom_normal_;

  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };
};


DECLARE_MAKER(SamplePlane)
  
SamplePlane::SamplePlane(GuiContext* ctx) : 
  Module("SamplePlane", ctx, Filter, "FieldsCreate", "SCIRun"),
  size_x_(ctx->subVar("sizex")),
  size_y_(ctx->subVar("sizey")),
  auto_size_(ctx->subVar("auto_size")),
  axis_(ctx->subVar("axis")),
  padpercent_(ctx->subVar("padpercent")),
  position_(ctx->subVar("pos")),
  data_at_(ctx->subVar("data-at")),
  update_type_(ctx->subVar("update_type")),
  custom_origin_(ctx->subVar("corigin")),
  custom_normal_(ctx->subVar("cnormal"))
{
}



SamplePlane::~SamplePlane()
{
}

void
SamplePlane::execute()
{
  update_state(NeedData);
  const int axis = Min(2, Max(0, axis_.get()));

  Transform trans;
  trans.load_identity();

  double angle = 0;
  Vector axis_vector(0.0, 0.0, 1.0);
  switch (axis)
  {
  case 0:
    angle = M_PI * -0.5; 
    axis_vector = Vector(0.0, 1.0, 0.0);
    break;

  case 1:
    angle = M_PI * 0.5; 
    axis_vector = Vector(1.0, 0.0, 0.0);
    break;

  case 2:
    angle = 0.0;
    axis_vector = Vector(0.0, 0.0, 1.0);
    break;

  default:
    break;
  }
  trans.pre_rotate(angle, axis_vector);

  if (axis_.get() == 3)
  {
    Vector tmp_normal(custom_normal_.get());
    Vector fakey(Cross(Vector(0.0, 0.0, 1.0), tmp_normal));
    if (fakey.length2() < 1.0e-6)
    {
      fakey = Cross(Vector(1.0, 0.0, 0.0), tmp_normal);
    }
    Vector fakex(Cross(tmp_normal, fakey));
    tmp_normal.safe_normalize();
    fakex.safe_normalize();
    fakey.safe_normalize();

    trans.load_identity();
    trans.load_basis(Point(0, 0, 0), fakex, fakey, tmp_normal);
    const Vector &origin(custom_origin_.get().asVector());
    trans.pre_translate(origin - fakex * 0.5 - fakey * 0.5);
  }

  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  DataTypeEnum datatype;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    datatype = SCALAR;
  }
  else
  {
    datatype = SCALAR;
    if (ifieldhandle->query_vector_interface(this).get_rep())
    {
      datatype = TENSOR;
    }
    else if (ifieldhandle->query_tensor_interface(this).get_rep())
    {
      datatype = VECTOR;
    }
  
    // Compute Transform.
    BBox box = ifieldhandle->mesh()->get_bounding_box();

    Vector diag(box.diagonal());
    trans.pre_scale(diag);

    if (axis_.get() != 3)
    {
      Point loc(box.min());
      position_.reset();
      double dist = position_.get()/2.0 + 0.5;
      switch (axis)
      {
      case 0:
        loc.x(loc.x() + diag.x() * dist);
        break;

      case 1:
        loc.y(loc.y() + diag.y() * dist);
        break;

      case 2:
        loc.z(loc.z() + diag.z() * dist);
        break;
      
      default:
        break;
      }

      trans.pre_translate(Vector(loc));
    }
  }
  
  unsigned int sizex, sizey;
  if( auto_size_.get() ){   // Guess at the size of the sample plane.
    // Currently we have only a simple algorithm for LatVolFields.
    if( LatVolMesh *lvm = dynamic_cast<LatVolMesh *> ((ifieldhandle->mesh()).get_rep()) ) {
      switch( axis ) {
      case 0:
        sizex = Max(2, (int)lvm->get_nj());
        size_x_.set( sizex );
        sizey = Max(2, (int)lvm->get_nk());
        size_y_.set( sizey );
        break;
      case 1: 
        sizex =  Max(2, (int)lvm->get_ni());
        size_x_.set( sizex );
        sizey =  Max(2, (int)lvm->get_nk());
        size_y_.set( sizey );
        break;
      case 2:
        sizex =  Max(2, (int)lvm->get_ni());
        size_x_.set( sizex );
        sizey =  Max(2, (int)lvm->get_nj());
        size_y_.set( sizey );
        break;
      default:
        warning("Custom axis, resize manually.");
        sizex = Max(2, size_x_.get());
        sizey = Max(2, size_y_.get());
        break;
      }
    } else {
      warning("No autosize algorithm for this field type, resize manually.");
      sizex = Max(2, size_x_.get());
      sizey = Max(2, size_y_.get());
    }
  } else {
    // Create blank mesh.
    sizex = Max(2, size_x_.get());
    sizey = Max(2, size_y_.get());
  }

  Point minb(0.0, 0.0, 0.0);
  Point maxb(1.0, 1.0, 0.0);
  Vector diag((maxb.asVector() - minb.asVector()) * (padpercent_.get()/100.0));
  minb -= diag;
  maxb += diag;

  ImageMeshHandle imagemesh = scinew ImageMesh(sizex, sizey, minb, maxb);

  int basis_order;
  if (data_at_.get() == "Nodes") basis_order = 1;
  else if (data_at_.get() == "Faces") basis_order = 0;
  else if (data_at_.get() == "None") basis_order = -1;
  else {
    error("Unsupported data_at location " + data_at_.get() + ".");
    return;
  }

  // Create Image Field.
  FieldHandle ofh;
  if (datatype == VECTOR)
  {
    ofh = scinew ImageField<Vector>(imagemesh, basis_order);
  }
  else if (datatype == TENSOR)
  {
    ofh = scinew ImageField<Tensor>(imagemesh, basis_order);
  }
  else
  {
    ofh = scinew ImageField<double>(imagemesh, basis_order);
  }

  // Transform field.
  ofh->mesh()->transform(trans);

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Sample Field");
  ofp->send(ofh);
}


} // End namespace SCIRun

