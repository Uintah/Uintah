
/*
 *  IntegrateCurrent: Compute current through a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class IntegrateCurrent : public Module {
  GuiDouble current_;
public:
  IntegrateCurrent(GuiContext *context);
  virtual ~IntegrateCurrent();
  virtual void execute();
};

DECLARE_MAKER(IntegrateCurrent)


IntegrateCurrent::IntegrateCurrent(GuiContext *context)
  : Module("IntegrateCurrent", context, Filter, "Forward", "BioPSE"),
    current_(context->subVar("current"))
{
}

IntegrateCurrent::~IntegrateCurrent()
{
}

void
IntegrateCurrent::execute()
{
  FieldIPort* efield_port = (FieldIPort *) get_iport("TetMesh EField");
  FieldIPort* sigmas_port = (FieldIPort *) get_iport("TetMesh Sigmas");
  FieldIPort* trisurf_port = (FieldIPort *) get_iport("TriSurf");
  if (!efield_port) {
    error("Unable to initialize iport 'TetMesh EField'.");
    return;
  }
  if (!sigmas_port) {
    error("Unable to initialize iport 'TetMesh Sigmas'.");
    return;
  }
  if (!trisurf_port) {
    error("Unable to initialize iport 'TriSurf'.");
    return;
  }
  
  FieldHandle efieldH, sigmasH, trisurfH;

  if (!efield_port->get(efieldH) || !efieldH.get_rep()) {
    error("Empty input E Field.");
    return;
  }
  if (!sigmas_port->get(sigmasH) || !sigmasH.get_rep()) {
    error("Empty input Sigmas.");
    return;
  }
  if (!trisurf_port->get(trisurfH) || !trisurfH.get_rep()) {
    error("Empty input trisurf.");
    return;
  }
  if (efieldH->mesh().get_rep() != sigmasH->mesh().get_rep()) {
    error("EField and Sigma Field need to have the same mesh.");
    return;
  }

  TetVolField<Vector> *efield = 
    dynamic_cast<TetVolField<Vector>*>(efieldH.get_rep());
  if (!efield) {
    error("EField isn't a TetVolField<Vector>.");
    return;
  }
  TetVolField<int> *sigmas = 
    dynamic_cast<TetVolField<int>*>(sigmasH.get_rep());
  if (!sigmas) {
    error("Sigmas isn't a TetVolField<int>.");
    return;
  }
  TriSurfMesh *tris = dynamic_cast<TriSurfMesh*>(trisurfH->mesh().get_rep());
  if (!tris) {
    error("Not a TriSurf.");
    return;
  }

  vector<pair<string, Tensor> > conds;
  if (!sigmasH->get_property("conductivity_table", conds)) {
    error("No conductivity_table found in Sigmas.");
    return;
  }
  int have_units = 0;
  string units;
  if (sigmasH->mesh()->get_property("units", units)) have_units=1;
  
  // for each face in tris, find its area, centroid, and normal
  // for that centroid, look up its sigma and efield in the tetvol fields
  // compute (sigma * efield * area) and dot it with the face normal
  // sum those up for all tris

  TriSurfMesh::Face::iterator fi, fe;
  tris->begin(fi);
  tris->end(fe);
  double current=0;
  TriSurfMesh::Node::array_type nodes;
  double total_area=0;
  while (fi != fe) {
    Point center;
    tris->get_center(center, *fi);
    double area = tris->get_area(*fi);
    total_area += area;
    tris->get_nodes(nodes, *fi);
    Point p0, p1, p2;
    tris->get_center(p0, nodes[0]);
    tris->get_center(p1, nodes[1]);
    tris->get_center(p2, nodes[2]);
    Vector normal(Cross(p2-p1,p2-p0));
    normal.normalize();
    TetVolMesh::Cell::index_type tet;
    if (!efield->get_typed_mesh()->locate(tet, center)) {
      error("Trisurf centroid was not located in tetvolmesh.");
      return;
    }
    Vector e = efield->value(tet);
    int sigma_idx = sigmas->value(tet);
    Tensor s(conds[sigma_idx].second);
    
    // compute sigma * e
    Vector c(s.mat_[0][0]*e.x()+s.mat_[0][1]*e.y()+s.mat_[0][2]*e.z(),
	     s.mat_[1][0]*e.x()+s.mat_[1][1]*e.y()+s.mat_[1][2]*e.z(),
	     s.mat_[2][0]*e.x()+s.mat_[2][1]*e.y()+s.mat_[2][2]*e.z());
    current += fabs(Dot(c,normal)) * area;
    ++fi;
  }
  if (have_units) {
    if (units == "mm") current/=1000;
    else if (units == "cm") current/=100;
    else if (units == "dm") current/=10;
    else warning("Unrecognized units '"  + units +"' will be ignored.");
  }
  current_.set(current);
}
} // End namespace BioPSE
