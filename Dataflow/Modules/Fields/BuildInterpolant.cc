/*
 *  BuildInterpolant.cc:  Build an interpolant field -- a field that says
 *         how to project the data from one field onto the data of a second
 *         field.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {

class BuildInterpolant : public Module
{
  FieldIPort  *source_field_;
  FieldIPort  *dest_field_;
  FieldOPort  *interp_field_;
  GuiString   interp_op_gui_;

public:
  BuildInterpolant(const clString& id);
  virtual ~BuildInterpolant();
  virtual void execute();
};

extern "C" Module* make_BuildInterpolant(const clString& id)
{
  return new BuildInterpolant(id);
}

BuildInterpolant::BuildInterpolant(const clString& id) : 
  Module("BuildInterpolant", id, Filter),
  interp_op_gui_("interp_op_gui", id, this)
{
  // Create the input ports
  source_field_ = scinew FieldIPort(this, "Source", FieldIPort::Atomic);
  add_iport(source_field_);
  dest_field_ = scinew FieldIPort(this, "Destination", FieldIPort::Atomic);
  add_iport(dest_field_);

  // Create the output port
  interp_field_ = scinew FieldOPort(this, "Interpolant", FieldIPort::Atomic);
  add_oport(interp_field_);
}

BuildInterpolant::~BuildInterpolant()
{
}

void BuildInterpolant::execute()
{
  FieldHandle sourceH;
  FieldHandle destH;

  if(!source_field_->get(sourceH))
    return;
  if(!dest_field_->get(destH))
    return;

  FieldHandle interpH;
  clString interp_op_gui = interp_op_gui_.get();

  // TODO: create the output field (clone dest Mesh), assign it to interpH
  //   based on the interpolation operator chosen in the Gui, build the
  //   interpolant map that maps data from the source Field onto the
  //   destination Mesh

  // ...

  interp_field_->send(interpH);
}

} // End namespace SCIRun
