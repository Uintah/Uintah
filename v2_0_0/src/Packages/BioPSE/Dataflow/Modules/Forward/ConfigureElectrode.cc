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
  GuiString active_;
  GuiDouble voltage_;
public:
  ConfigureElectrode(GuiContext *context);
  virtual ~ConfigureElectrode();
  virtual void execute();
};

DECLARE_MAKER(ConfigureElectrode)

ConfigureElectrode::ConfigureElectrode(GuiContext *context)
  : Module("ConfigureElectrode", context, Filter, "Forward", "BioPSE"),
    active_(context->subVar("active")), voltage_(context->subVar("voltage"))
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
  string active = active_.get();

  CurveMesh::Node::iterator ni;
  elecFld->get_typed_mesh()->begin(ni);
  elecFld->fdata()[*ni]=voltage;

  elecFld->set_property("active_side", active, false);
  oelec->send(elecFld);
}
} // End namespace BioPSE
