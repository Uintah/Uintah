/*
 *  SetProperty: Set a property for a Field (or its Mesh)
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class SetProperty : public Module {
  GuiString prop_;
  GuiString val_;
  GuiInt mesh_prop_; // is this property for the mesh or the Field?
public:
  SetProperty(const string& id);
  virtual ~SetProperty();
  virtual void execute();
};

extern "C" Module* make_SetProperty(const string& id)
{
    return new SetProperty(id);
}

SetProperty::SetProperty(const string& id)
: Module("SetProperty", id, Filter,"Fields", "SCIRun"),
  prop_("prop", id, this), val_("val", id, this),
  mesh_prop_("meshprop", id, this)
{
}

SetProperty::~SetProperty()
{
}

void SetProperty::execute() {
  FieldIPort *ifield = (FieldIPort *)get_iport("Input");
  FieldOPort *ofield = (FieldOPort *)get_oport("Output");

  if (!ifield) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ofield) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  FieldHandle fldH;
  if (!ifield->get(fldH))
    return;
  if (!fldH.get_rep()) {
    warning("Empty input field.");
    return;
  }

  fldH->generation++;

  // set this new property
  if (mesh_prop_.get()) {
    fldH->mesh()->generation++;
    fldH->mesh()->store(prop_.get(), val_.get(), false);
  } else 
    fldH->store(prop_.get(), val_.get(), false);
  
  ofield->send(fldH);
}
} // End namespace SCIRun
