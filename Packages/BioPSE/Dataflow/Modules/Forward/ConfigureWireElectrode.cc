/*
 *  ConfigureWireElectrode: Insert an electrode into a finite element mesh
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class ConfigureWireElectrode : public Module {
  GuiDouble voltage_;
  GuiDouble radius_;
  GuiInt nu_;
public:
  ConfigureWireElectrode(GuiContext *context);
  virtual ~ConfigureWireElectrode();
  virtual void execute();
};

DECLARE_MAKER(ConfigureWireElectrode)

ConfigureWireElectrode::ConfigureWireElectrode(GuiContext *context)
  : Module("ConfigureWireElectrode", context, Filter, "Forward", "BioPSE"),
    voltage_(context->subVar("voltage")), radius_(context->subVar("radius")),
    nu_(context->subVar("nu"))
{
}

ConfigureWireElectrode::~ConfigureWireElectrode()
{
}

void ConfigureWireElectrode::execute() {
  FieldIPort* ielec = (FieldIPort *) get_iport("Electrode");
  FieldOPort* oelec = (FieldOPort *) get_oport("Electrode");
  
  if (!ielec) {
    error("Unable to initialize iport 'Electrode'.");
    return;
  }
  if (!oelec) {
    error("Unable to initialize oport 'Electrode'.");
    return;
  }
  
  FieldHandle ielecH;

  if (!ielec->get(ielecH))
    return;
  if (!ielecH.get_rep()) {
    error("Empty input electrode.");
    return;
  }
  MeshHandle meshH = ielecH->mesh();
  CurveMesh *mesh=dynamic_cast<CurveMesh *>(meshH.get_rep());
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


  CurveMesh::Node::iterator ni, ne;
  CurveMesh::Node::size_type nn;
  mesh->begin(ni);  
  mesh->end(ne);
  mesh->size(nn);
  QuadSurfMeshHandle quadMesh = new QuadSurfMesh;
  Array1<Array1<Point> > pts(nn);
  Array1<Array1<QuadSurfMesh::Node::index_type> > pts_idx(nn);
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
      up=(c[i+1]-c[i]).normal();
    } else if (i==(unsigned int)(nn-1)) {
      up=(c[i]-c[i-1]).normal();
    } else {
      up=(c[i+1]-c[i-1]).normal();
    }
    if (i==0) {
      up.find_orthogonal(b1, b2);
    } else {
      b2=(Cross(up, b1)).normal();
      b1=(Cross(b2, up)).normal();
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
  QuadSurfField<double>* quadFld = new QuadSurfField<double>(quadMesh, 
							     Field::NODE);
  QuadSurfMesh::Node::iterator qni, qne;
  quadMesh->begin(qni);
  quadMesh->end(qne);
  while(qni != qne) {
    quadFld->set_value(voltage, *qni);
    ++qni;
  }
  oelec->send(quadFld);
}
} // End namespace BioPSE
