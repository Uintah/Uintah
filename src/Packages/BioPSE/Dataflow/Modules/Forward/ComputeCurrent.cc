
/*
 *  ComputeCurrent: Compute current through a surface
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

class ComputeCurrent : public Module {
  GuiDouble current_;
public:
  ComputeCurrent(const string& id);
  virtual ~ComputeCurrent();
  virtual void execute();
};

extern "C" Module* make_ComputeCurrent(const string& id)
{
    return new ComputeCurrent(id);
}

ComputeCurrent::ComputeCurrent(const string& id)
  : Module("ComputeCurrent", id, Filter, "Forward", "BioPSE"),
    current_("current", id, this)
{
}

ComputeCurrent::~ComputeCurrent()
{
}

void ComputeCurrent::execute() {
  FieldIPort* efield_port = (FieldIPort *) get_iport("TetMesh EField");
  FieldIPort* sigmas_port = (FieldIPort *) get_iport("TetMesh Sigmas");
  FieldIPort* trisurf_port = (FieldIPort *) get_iport("TriSurf");
  if (!efield_port) {
    postMessage("Unable to initialize "+name+"'s TetMesh EField port\n");
    return;
  }
  if (!sigmas_port) {
    postMessage("Unable to initialize "+name+"'s TetMesh Sigmas port\n");
    return;
  }
  if (!trisurf_port) {
    postMessage("Unable to initialize "+name+"'s TriSurf port\n");
    return;
  }
  
  FieldHandle efieldH, sigmasH, trisurfH;

  if (!efield_port->get(efieldH) || !efieldH.get_rep()) {
    cerr << "ComputeCurrent: error - empty input E Field.\n";
    return;
  }
  if (!sigmas_port->get(sigmasH) || !sigmasH.get_rep()) {
    cerr << "ComputeCurrent: error - empty input Sigmas.\n";
    return;
  }
  if (!trisurf_port->get(trisurfH) || !trisurfH.get_rep()) {
    cerr << "ComputeCurrent: error - empty input trisurf.\n";
    return;
  }
  if (efieldH->mesh().get_rep() != sigmasH->mesh().get_rep()) {
    cerr << "ComputeCurrent: error - EField and Sigma Field need to have the same mesh.\n";
    return;
  }

  TetVolField<Vector> *efield = 
    dynamic_cast<TetVolField<Vector>*>(efieldH.get_rep());
  if (!efield) {
    cerr << "ComputeCurrent: error - EField isn't a TetVolField<Vector>.\n";
    return;
  }
  TetVolField<int> *sigmas = 
    dynamic_cast<TetVolField<int>*>(sigmasH.get_rep());
  if (!sigmas) {
    cerr << "ComputeCurrent: error - Sigmas isn't a TetVolField<int>.\n";
    return;
  }
  TriSurfMesh *tris = dynamic_cast<TriSurfMesh*>(trisurfH->mesh().get_rep());
  if (!tris) {
    cerr << "ComputeCurrent: error - not a TriSurf.\n";
    return;
  }

  vector<pair<string, Tensor> > conds;
  if (!sigmasH->get_property("conductivity_table", conds)) {
    cerr << "ComputeCurrent: error - no conductivity_table found in Sigmas.\n";
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
  while (fi != fe) {
    Point center;
    tris->get_center(center, *fi);
    double area = tris->get_area(*fi);
    tris->get_nodes(nodes, *fi);
    Point p0, p1, p2;
    tris->get_center(p0, nodes[0]);
    tris->get_center(p1, nodes[1]);
    tris->get_center(p2, nodes[2]);
    Vector normal(Cross(p2-p0,p2-p1));
    normal.normalize();
    TetVolMesh::Cell::index_type tet;
    if (!efield->get_typed_mesh()->locate(tet, center)) {
      cerr << "ComputeCurrent: error - trisurf centroid was not located in tetvolmesh.\n";
      return;
    }
    Vector e = efield->value(tet);
    int sigma_idx = sigmas->value(tet);
    Tensor s(conds[sigma_idx].second);
    
    // compute sigma * e
    Vector c(s.mat_[0][0]*e.x()+s.mat_[0][1]*e.y()+s.mat_[0][2]*e.z(),
	     s.mat_[1][0]*e.x()+s.mat_[1][1]*e.y()+s.mat_[1][2]*e.z(),
	     s.mat_[2][0]*e.x()+s.mat_[2][1]*e.y()+s.mat_[2][2]*e.z());
    current += Dot(c,normal) * area;
    ++fi;
  }
  if (have_units) {
    if (units == "mm") current/=1000;
    else if (units == "cm") current/=100;
    else if (units == "dm") current/=10;
    else cerr << "ComputeCurrent: warning - unrecognized units "<<units<<" will be ignored.\n";
  }
  current_.set(current);
}
} // End namespace BioPSE
