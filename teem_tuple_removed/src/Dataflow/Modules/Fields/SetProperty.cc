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
  SetProperty(GuiContext* ctx);
  virtual ~SetProperty();
  virtual void execute();
};

DECLARE_MAKER(SetProperty)
SetProperty::SetProperty(GuiContext* ctx)
: Module("SetProperty", ctx, Filter,"FieldsOther", "SCIRun"),
  prop_(ctx->subVar("prop")), val_(ctx->subVar("val")),
  mesh_prop_(ctx->subVar("meshprop"))
{
}

SetProperty::~SetProperty()
{
}

void SetProperty::execute() {
  FieldIPort *ifield = (FieldIPort *)get_iport("Input");
  FieldOPort *ofield = (FieldOPort *)get_oport("Output");

  if (!ifield) {
    error("Unable to initialize iport 'Input'.");
    return;
  }
  if (!ofield) {
    error("Unable to initialize oport 'Output'.");
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
    fldH->mesh()->set_property(prop_.get(), val_.get(), false);
  } else 
    fldH->set_property(prop_.get(), val_.get(), false);
  
  ofield->send(fldH);
}
} // End namespace SCIRun
