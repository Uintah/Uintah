/*
 *  GenField.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/LatticeVol.h>
#include <math.h>

#include <Packages/Moulding/share/share.h>

namespace Moulding {

using namespace SCIRun;

class MouldingSHARE GenField : public Module {
public:
  GenField(const clString& id);

  virtual ~GenField();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

private:
  FieldOPort* oport;
};

extern "C" MouldingSHARE Module* make_GenField(const clString& id) {
  return new GenField(id);
}

GenField::GenField(const clString& id)
  : Module("GenField", id, Source)
{
  oport = scinew FieldOPort(this, "Sample Field", FieldIPort::Atomic);
  add_oport(oport);
}

GenField::~GenField(){
}

void GenField::execute()
{
  LatticeVol<Vector> *vf = scinew LatticeVol<Vector>(Field::NODE);
  LatVolMesh* mesh = dynamic_cast<LatVolMesh*>(vf->get_typed_mesh().get_rep());
  LatVolMesh::node_size_type size(64,64,64);
  LatticeVol<Vector>::fdata_type &fdata = vf->fdata();
  mesh->set_nx(64);
  mesh->set_ny(64);
  mesh->set_nz(64);
  mesh->set_min(Point(-31.4,-30.0,-10.0));
  mesh->set_max(Point(31.4,30.0,10.0));
  fdata.resize(size);

  LatVolMesh::node_iterator ni = mesh->node_begin();
  LatVolMesh::node_index i;
  Point p;
  for (;ni!=mesh->node_end();++ni) {
    i = *ni;
    mesh->get_point(p,i);
    fdata[i] = Vector(0.05,0.05*sin(.5*p.x()),0);
  }

  oport->send(vf);
}

void GenField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Moulding


