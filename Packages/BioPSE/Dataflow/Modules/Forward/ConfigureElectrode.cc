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

#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/TetVol.h>
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
  ConfigureElectrode(const string& id);
  virtual ~ConfigureElectrode();
  virtual void execute();
};

extern "C" Module* make_ConfigureElectrode(const string& id)
{
    return new ConfigureElectrode(id);
}

ConfigureElectrode::ConfigureElectrode(const string& id)
  : Module("ConfigureElectrode", id, Filter, "Forward", "BioPSE"),
    flip_("flip", id, this), voltage_("voltage", id, this)
{
}

ConfigureElectrode::~ConfigureElectrode()
{
}

void ConfigureElectrode::execute() {
  FieldIPort* ielec = (FieldIPort *) get_iport("Electrode");
  FieldOPort* oelec = (FieldOPort *) get_oport("Electrode");
  
  if (!ielec) {
    postMessage("Unable to initialize "+name+"'s ielec port\n");
    return;
  }
  if (!oelec) {
    postMessage("Unable to initialize "+name+"'s oelec port\n");
    return;
  }
  
  FieldHandle ielecH;

  if (!ielec->get(ielecH))
    return;
  if (!ielecH.get_rep()) {
    cerr << "ConfigureElectrode: error - empty input electrode.\n";
    return;
  }
  ContourField<double> *elecFld = dynamic_cast<ContourField<double>*>(ielecH.get_rep());
  if (!elecFld) {
    cerr << "ConfigureElectrode: error - input electrode wasn't a ContourField<double>\n";
    return;
  }

  double voltage = voltage_.get();
  int flip = flip_.get();
  ContourMesh::Node::iterator ni, ne;
  elecFld->get_typed_mesh()->begin(ni);
  elecFld->fdata()[*ni]=voltage;

  if (!flip) {
    oelec->send(ielecH);
    return;
  }

  ContourMeshHandle elecMesh(scinew ContourMesh(*(elecFld->get_typed_mesh().get_rep())));
  ContourField<double>* elecFldFlip = scinew ContourField<double>(elecMesh, Field::NODE);

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
