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
 *  SampleLattice.cc:  Make an ImageField that fits the source field.
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
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

class SampleLattice : public Module
{
public:
  SampleLattice(GuiContext* ctx);
  virtual ~SampleLattice();

  virtual void execute();

private:
  GuiInt size_x_;
  GuiInt size_y_;
  GuiInt size_z_;
  GuiDouble padpercent_;
  GuiString data_at_;

  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };
};


DECLARE_MAKER(SampleLattice)

SampleLattice::SampleLattice(GuiContext* ctx)
  : Module("SampleLattice", ctx, Filter, "FieldsCreate", "SCIRun"),
    size_x_(ctx->subVar("sizex")),
    size_y_(ctx->subVar("sizey")),
    size_z_(ctx->subVar("sizez")),
    padpercent_(ctx->subVar("padpercent")),
    data_at_(ctx->subVar("data-at"))
{
}



SampleLattice::~SampleLattice()
{
}


void
SampleLattice::execute()
{
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;

  Point minb, maxb;
  DataTypeEnum datatype;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
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
  LatVolMeshHandle mesh = scinew LatVolMesh(sizex, sizey, sizez, minb, maxb);

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
    LatVolField<double> *lvf = scinew LatVolField<double>(mesh, basis_order);
    if (basis_order != -1)
    {
      LatVolField<double>::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = 0.0;
	++itr;
      }    
    }
    ofh = lvf;
  } 
  else if (datatype == VECTOR)
  {
    LatVolField<Vector> *lvf = scinew LatVolField<Vector>(mesh, basis_order);
    if (basis_order != -1)
    {
      LatVolField<Vector>::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = Vector(0.0, 0.0, 0.0);
	++itr;
      }
    }
    ofh = lvf;
  }				    
  else // if (datatype == TENSOR)	    
  {				    
    LatVolField<Tensor> *lvf = scinew LatVolField<Tensor>(mesh, basis_order);
    if (basis_order != -1)
    {
      LatVolField<Tensor>::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = Tensor(0.0);
	++itr;
      }
    }
    ofh = lvf;
  }				    

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Sample Field");
  ofp->send(ofh);
}


} // End namespace SCIRun

