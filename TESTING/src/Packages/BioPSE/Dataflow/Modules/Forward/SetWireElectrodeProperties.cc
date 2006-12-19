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

/*
 *  SetWireElectrodeProperties: Insert an electrode into a finite element mesh
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;
typedef CurveMesh<CrvLinearLgn<Point> >            CMesh;
typedef QuadSurfMesh<QuadBilinearLgn<Point> >      QSMesh;
typedef QuadBilinearLgn<double>                    QFBasis;
typedef GenericField<QSMesh, QFBasis, vector<double> > QSField; 

class SetWireElectrodeProperties : public Module {
  GuiDouble voltage_;
  GuiDouble radius_;
  GuiInt nu_;
public:
  SetWireElectrodeProperties(GuiContext *context);
  virtual ~SetWireElectrodeProperties();
  virtual void execute();
};

DECLARE_MAKER(SetWireElectrodeProperties)

SetWireElectrodeProperties::SetWireElectrodeProperties(GuiContext *context)
  : Module("SetWireElectrodeProperties", context, Filter, "Forward", "BioPSE"),
    voltage_(context->subVar("voltage")), radius_(context->subVar("radius")),
    nu_(context->subVar("nu"))
{
}


SetWireElectrodeProperties::~SetWireElectrodeProperties()
{
}


void
SetWireElectrodeProperties::execute()
{
  FieldHandle ielecH;
  if (!get_input_handle("Electrode", ielecH)) return;

  MeshHandle meshH = ielecH->mesh();
  CMesh *mesh=dynamic_cast<CMesh *>(meshH.get_rep());
  if (!mesh) {
    error("Input electrode wasn't a CurveField.");
    return;
  }

  double voltage = voltage_.get();
  double radius = radius_.get();
  int nu = nu_.get();

  if (radius < 0) {
    error("Radius can't be negative");
    return;
  }
  if (nu < 3) {
    error("NU can't be less than 3");
    return;
  }
  double du=M_PI*2./nu;

  CMesh::Node::iterator ni, ne;
  CMesh::Node::size_type nn;
  mesh->begin(ni);  
  mesh->end(ne);
  mesh->size(nn);
  QSMesh::handle_type quadMesh = new QSMesh;
  Array1<Array1<Point> > pts(nn);
  Array1<Array1<QSMesh::Node::index_type> > pts_idx(nn);
  if (nn < 2) {
    error("Need at least two points along Curve");
    return;
  }
  Array1<Point> c(nn);
  int idx=0;
  while(ni != ne) {
    mesh->get_center(c[idx], *ni);
    ++ni;
    ++idx;
  }
  Vector up, b1, b2;
  for (unsigned int i=0; i<nn; i++) {
    pts[i].resize(nu);
    pts_idx[i].resize(nu);
    if (i==0) {
      up=(c[i+1]-c[i]).safe_normal();
    } else if (i==(unsigned int)(nn-1)) {
      up=(c[i]-c[i-1]).safe_normal();
    } else {
      up=(c[i+1]-c[i-1]).safe_normal();
    }
    if (i==0) {
      up.find_orthogonal(b1, b2);
    } else {
      b2=(Cross(up, b1)).safe_normal();
      b1=(Cross(b2, up)).safe_normal();
    }
    for (int u=0; u<nu; u++) {
      pts[i][u]=c[i]+(b1*cos(u*du)+b2*sin(u*du))*radius;
      pts_idx[i][u]=quadMesh->add_point(pts[i][u]);
    }
  }
  for (unsigned int i=0; i<(unsigned int)(nn-1); i++) {
    int u;
    for (u=0; u<nu-1; u++) {
      quadMesh->add_quad(pts[i][u], pts[i][u+1], pts[i+1][u+1], pts[i+1][u]);
    }
    quadMesh->add_quad(pts[i][u], pts[i][0], pts[i+1][0], pts[i+1][u]);
  }
  QSField* quadFld = new QSField(quadMesh);
  QSMesh::Node::iterator qni, qne;
  quadMesh->begin(qni);
  quadMesh->end(qne);
  while(qni != qne) {
    quadFld->set_value(voltage, *qni);
    ++qni;
  }
  FieldHandle qfield(quadFld);
  send_output_handle("Electrode", qfield);
}

} // End namespace BioPSE
