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
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/FData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/GenericField.h>
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
  GuiInt axis_;
  GuiDouble padpercent_;
  GuiDouble position_;
  GuiString data_at_;
  GuiString update_type_;

  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };
};


DECLARE_MAKER(SamplePlane)
  
SamplePlane::SamplePlane(GuiContext* ctx) : 
  Module("SamplePlane", ctx, Filter, "FieldsCreate", "SCIRun"),
  size_x_(ctx->subVar("sizex")),
  size_y_(ctx->subVar("sizey")),
  axis_(ctx->subVar("axis")),
  padpercent_(ctx->subVar("padpercent")),
  position_(ctx->subVar("pos")),
  data_at_(ctx->subVar("data-at")),
  update_type_(ctx->subVar("update_type"))
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

  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
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

    Point loc(box.min());
    Vector diag(box.diagonal());
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

    trans.pre_scale(diag);
    trans.pre_translate(Vector(loc));
  }
  
  // Create blank mesh.
  unsigned int sizex = Max(2, size_x_.get());
  unsigned int sizey = Max(2, size_y_.get());
  Point minb(0.0, 0.0, 0.0);
  Point maxb(1.0, 1.0, 1.0);
  Vector diag((maxb.asVector() - minb.asVector()) * (padpercent_.get()/100.0));
  minb -= diag;
  maxb += diag;
  
  typedef ImageMesh<QuadBilinearLgn<Point> > IMesh;
  IMesh::handle_type imagemesh = scinew IMesh(sizex, sizey, minb, maxb);

  int basis_order;
  if (data_at_.get() == "Nodes") basis_order = 1;
  else if (data_at_.get() == "Faces") basis_order = 0;
  else if (data_at_.get() == "None") basis_order = -1;
  else {
    error("Unsupported data_at location " + data_at_.get() + ".");
    return;
  }

  if (data_at_.get() == "Faces") basis_order = 0;
  else basis_order = 1;

  // Create Image Field.
  FieldHandle ofh;
  if (datatype == VECTOR)
  {
    typedef NoDataBasis<Vector>                 NBasis;
    typedef ConstantBasis<Vector>               CBasis;
    typedef QuadBilinearLgn<Vector>             LBasis;

    if (basis_order == -1) {
      typedef GenericField<IMesh, NBasis, FData2d<Vector, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    } else if (basis_order == 0) {
      typedef GenericField<IMesh, CBasis, FData2d<Vector, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    } else {
      typedef GenericField<IMesh, LBasis, FData2d<Vector, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    }
  }
  else if (datatype == TENSOR)
  {
    typedef NoDataBasis<Tensor>                      NBasis;
    typedef ConstantBasis<Tensor>               CBasis;
    typedef QuadBilinearLgn<Tensor>             LBasis;

    if (basis_order == -1) {
      typedef GenericField<IMesh, NBasis, FData2d<Tensor, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    } else if (basis_order == 0) {
      typedef GenericField<IMesh, CBasis, FData2d<Tensor, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    } else {
      typedef GenericField<IMesh, LBasis, FData2d<Tensor, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    }
  }
  else
  {
    typedef NoDataBasis<double>                      NBasis;
    typedef ConstantBasis<double>               CBasis;
    typedef QuadBilinearLgn<double>             LBasis;

    if (basis_order == -1) {
      typedef GenericField<IMesh, NBasis, FData2d<double, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    } else if (basis_order == 0) {
      typedef GenericField<IMesh, CBasis, FData2d<double, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    } else {
      typedef GenericField<IMesh, LBasis, FData2d<double, IMesh> > IField;
      IField *lvf = scinew IField(imagemesh);
      ofh = lvf;
    }
  }

  // Transform field.
  ofh->mesh()->transform(trans);

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Sample Field");
  if (!ofp) {
    error("Unable to initialize oport 'Output Sample Field'.");
    return;
  }
  ofp->send(ofh);
}


} // End namespace SCIRun

