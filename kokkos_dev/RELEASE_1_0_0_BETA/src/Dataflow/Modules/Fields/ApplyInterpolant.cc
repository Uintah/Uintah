/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  ApplyInterpolant.cc:  Apply an interpolant field to project the data
 *                 from one field onto the mesh of another field.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {

class ApplyInterpolant : public Module {
public:
  ApplyInterpolant(const clString& id);
  virtual ~ApplyInterpolant();
  virtual void execute();
private:
  FieldIPort  *source_field_;
  FieldIPort  *dest_field_;
  FieldIPort  *interp_field_;
  FieldOPort  *output_field_;
};

extern "C" Module* make_ApplyInterpolant(const clString& id)
{
  return new ApplyInterpolant(id);
}

ApplyInterpolant::ApplyInterpolant(const clString& id)
  : Module("ApplyInterpolant", id, Filter)
{
  // Create the input ports
  source_field_ = scinew FieldIPort(this, "Source", FieldIPort::Atomic);
  add_iport(source_field_);
  dest_field_ = scinew FieldIPort(this, "Destination", FieldIPort::Atomic);
  add_iport(dest_field_);
  interp_field_ = scinew FieldIPort(this, "Interpolant", FieldIPort::Atomic);
  add_iport(interp_field_);

  // Create the output port
  output_field_ = scinew FieldOPort(this, "Output", FieldIPort::Atomic);
  add_oport(output_field_);
}

ApplyInterpolant::~ApplyInterpolant()
{
}

void ApplyInterpolant::execute()
{
  FieldHandle sourceH;
  FieldHandle destH;
  FieldHandle interpH;

  if(!source_field_->get(sourceH))
    return;
  if(!dest_field_->get(destH))
    return;
  if(!interp_field_->get(interpH))
    return;

  FieldHandle outputH;

  // TODO: create the output field (clone dest Mesh), assign it to outputH
  //   make sure the dimensions/locations of source/dest/interp all line up
  //   map the data from source, through interp, into output

  // ...

  output_field_->send(outputH);
}

} // End namespace SCIRun


