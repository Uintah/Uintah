
/*
 *  ComputeCurrent: Compute current through a volume
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

public:
  ComputeCurrent(GuiContext *context);
  virtual ~ComputeCurrent();
  virtual void execute();
};

DECLARE_MAKER(ComputeCurrent)


ComputeCurrent::ComputeCurrent(GuiContext *context)
  : Module("ComputeCurrent", context, Filter, "Forward", "BioPSE")
{
}

ComputeCurrent::~ComputeCurrent()
{
}

void
ComputeCurrent::execute()
{
  FieldIPort* efield_port = (FieldIPort *) get_iport("TetMesh EField");
  FieldIPort* sigmas_port = (FieldIPort *) get_iport("TetMesh Sigmas");
  FieldOPort* ofield_port = (FieldOPort *) get_oport("Currents");

  if (!efield_port) {
    error("Unable to initialize iport 'TetMesh EField'.");
    return;
  }
  if (!sigmas_port) {
    error("Unable to initialize iport 'TetMesh Sigmas'.");
    return;
  }
  if (!ofield_port) {
    error("Unable to initialize oport 'Currents'.");
    return;
  }

  FieldHandle efieldH, sigmasH;

  if (!efield_port->get(efieldH) || !efieldH.get_rep()) {
    error("Empty input E Field.");
    return;
  }
  if (!sigmas_port->get(sigmasH) || !sigmasH.get_rep()) {
    error("Empty input Sigmas.");
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
  bool index_based = true;
  TetVolField<int> *sigmasInt = 
    dynamic_cast<TetVolField<int>*>(sigmasH.get_rep());
  TetVolField<Tensor> *sigmasTensor =
    dynamic_cast<TetVolField<Tensor>*>(sigmasH.get_rep());
  if (!sigmasInt && !sigmasTensor) {
    error("Sigmas isn't a TetVolField<Tensor> or TetVolField<int>.");
    return;
  }
  if (sigmasTensor) index_based = false;

  if (sigmasH->data_at() != Field::CELL) {
    error("Need sigmas at Cells");
    return;
  }
  if (efieldH->data_at() != Field::CELL) {
    error("Need efield at Cells");
    return;
  }

  vector<pair<string, Tensor> > conds;
  if (index_based && !sigmasH->get_property("conductivity_table", conds)) {
    error("No conductivity_table found in Sigmas.");
    return;
  }
  int have_units = 0;
  string units;
  if (sigmasH->mesh()->get_property("units", units)) have_units=1;

  // For each cell in the mesh, find the dot product of the gradient
  // vector and the conductivity tensor.  The result is a vector field
  // with data at cells.

  // Create output mesh
  //  OFIELD *ofield = scinew OFIELD(imesh, Field::CELL);
  
  TetVolMeshHandle mesh = efield->get_typed_mesh();
  TetVolMesh::Cell::iterator fi, fe;
  mesh->begin(fi);
  mesh->end(fe);

  TetVolField<Vector> *ofield = new TetVolField<Vector>(mesh, Field::CELL);

  while (fi != fe) {
    Vector vec;
    Vector e;
    efield->value(e, *fi);
    Tensor s;
    if (index_based) {
      int sigma_idx;
      sigmasInt->value(sigma_idx, *fi);
      s=conds[sigma_idx].second;
    } else {
      sigmasTensor->value(s, *fi);
    }
    // - sign added to vector to account for E = - Del V
    vec = Vector(-(s.mat_[0][0]*e.x()+s.mat_[0][1]*e.y()+s.mat_[0][2]*e.z()),
		 -(s.mat_[1][0]*e.x()+s.mat_[1][1]*e.y()+s.mat_[1][2]*e.z()),
		 -(s.mat_[2][0]*e.x()+s.mat_[2][1]*e.y()+s.mat_[2][2]*e.z()));

    if (have_units) {
      if (units == "mm") vec/=1000;
      else if (units == "cm") vec/=100;
      else if (units == "dm") vec/=10;
      else warning("Unrecognized units '"  + units +"' will be ignored.");
    }
    ofield->set_value(vec,*fi);
    ++fi;
  }
  ofield_port->send(ofield);
}
} // End namespace BioPSE
