/*
 *  ManageFieldSet: Manage the members of a field set -- 
 *                                  create, add, delete, extract
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/FieldSet.h>
#include <Dataflow/Ports/FieldSetPort.h>

#include <iostream>
using std::cerr;
#include <stdio.h>


namespace SCIRun {

class ManageFieldSet : public Module
{
  GuiString          op_gui_;


public:
  ManageFieldSet(const clString& id);
  virtual ~ManageFieldSet();
  virtual void execute();
};


extern "C" Module* make_ManageFieldSet(const clString& id)
{
  return new ManageFieldSet(id);
}


ManageFieldSet::ManageFieldSet(const clString& id)
  : Module("ManageFieldSet", id, Filter),
    op_gui_("op_gui", id, this)
{
}


ManageFieldSet::~ManageFieldSet()
{
}


void
ManageFieldSet::execute()
{
  update_state(NeedData);

  clString op_gui = op_gui_.get();

  // FIXME: based on the op_gui variable, do something to a FieldSet

  // ...

}

} // End namespace SCIRun
