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

#include <Core/Containers/Array1.h>
#include <Core/Containers/String.h>
#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/Expon.h>

#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {


class ManageFieldSet : public Module
{
  FieldIPort         *isurf_;
  MatrixIPort *imat_;
  GuiString          surfid_;

  FieldOPort      *osurf_;

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
  : Module("ManageFieldSet", id, Filter), surfid_("surfid", id, this)
{
  isurf_ = new FieldIPort(this, "SurfIn", FieldIPort::Atomic);
  add_iport(isurf_);
  imat_ = new MatrixIPort(this, "MatIn", MatrixIPort::Atomic);
  add_iport(imat_);

  // Create the output port
  osurf_ = new FieldOPort(this, "SurfOut", FieldIPort::Atomic);
  add_oport(osurf_);
}

ManageFieldSet::~ManageFieldSet()
{
}

void
ManageFieldSet::execute()
{
  update_state(NeedData);

  FieldHandle sh;
  if (!isurf_->get(sh))
  {
    return;
  }
  //TriSurfMesh *ts = sh.get_rep();  // FIXME: extract surf from field
  TriSurfMesh *ts = 0;
  if (!ts)
  {
    cerr << "Error: surface isn't a TriSurfMesh\n";
    return;
  }

  update_state(JustStarted);

  MatrixHandle cmh;
  if (!imat_->get(cmh)) return;
  if (!cmh.get_rep())
  {
    cerr << "Error: empty matrix\n";
    return;
  }

  // TODO
  // convert imat_ into an indexed attribute.
  // Make new field with same geometry, imat_ indexed data set.


  //osurf_->send(sh);  // TODO: fix this, send surface field
}

} // End namespace SCIRun
