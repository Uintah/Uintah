/*
 *  ConfigureElectrode: Insert an electrode into a finite element mesh
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
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class ConfigureElectrode : public Module {
  GuiInt flip_;
  GuiDouble voltage_;
public:
  ConfigureElectrode(GuiContext *context);
  virtual ~ConfigureElectrode();
  virtual void execute();
};

DECLARE_MAKER(ConfigureElectrode)

ConfigureElectrode::ConfigureElectrode(GuiContext *context)
  : Module("ConfigureElectrode", context, Filter, "Forward", "BioPSE"),
    flip_(context->subVar("flip")),
    voltage_(context->subVar("voltage"))
{
}

ConfigureElectrode::~ConfigureElectrode()
{
}

void ConfigureElectrode::execute() {
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
  CurveField<double> *elecFld = dynamic_cast<CurveField<double>*>(ielecH.get_rep());
  if (!elecFld) {
    error("Input electrode wasn't a CurveField<double>.");
    return;
  }

  double voltage = voltage_.get();
  int flip = flip_.get();
  CurveMesh::Node::iterator ni, ne;
  elecFld->get_typed_mesh()->begin(ni);
  elecFld->fdata()[*ni]=voltage;

  if (!flip) {
    oelec->send(ielecH);
    return;
  }

  CurveMeshHandle elecMesh(scinew CurveMesh(*(elecFld->get_typed_mesh().get_rep())));
  CurveField<double>* elecFldFlip = scinew CurveField<double>(elecMesh, Field::NODE);

  elecMesh->begin(ni);
  elecMesh->end(ne);
  Point pt;
  elecMesh->get_center(pt, *ni);
  double midX = pt.x();
  double midY = pt.y();
  while(ni != ne) {
    Point pt;
    elecMesh->get_center(pt, *ni);
    pt.x(2*midX-pt.x());
    pt.y(2*midY-pt.y());
    elecMesh->set_point(pt, *ni);
    elecFldFlip->fdata()[*ni] = elecFld->fdata()[*ni];
    ++ni;
  }
  oelec->send(elecFldFlip);
}
} // End namespace BioPSE
