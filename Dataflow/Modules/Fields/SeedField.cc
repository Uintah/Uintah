/*
 *  SeedField.cc:  From a mesh, seed some number of dipoles
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   October 2000
 *
 *  Copyright (C) 2000 SCI Group
 */
 
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
using std::cerr;

namespace SCIRun {

class SeedField : public Module {
  MeshIPort* imesh;
  MatrixOPort* omat;  
  GuiString seedRandTCL;
  GuiString numDipolesTCL;
  GuiString dipoleMagnitudeTCL;
public:
  SeedField(const clString& id);
  virtual ~SeedField();
  virtual void execute();
};

extern "C" Module* make_SeedField(const clString& id) {
  return new SeedField(id);
}

SeedField::SeedField(const clString& id)
  : Module("SeedField", id, Filter), seedRandTCL("seedRandTCL", id, this),
  numDipolesTCL("numDipolesTCL", id, this),
  dipoleMagnitudeTCL("dipoleMagnitudeTCL", id, this)
{
  // Create the input port
  imesh = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
  add_iport(imesh);
  
  // Create the output ports
  omat=scinew MatrixOPort(this,"Dipole Seeds", MatrixIPort::Atomic);
  add_oport(omat);
}

SeedField::~SeedField()
{
}

void SeedField::execute()
{
  MeshHandle mesh;
  if (!imesh->get(mesh) || !mesh.get_rep()) return;
  int seedRand, numDipoles;
  double dipoleMagnitude;

  seedRandTCL.get().get_int(seedRand);
  numDipolesTCL.get().get_int(numDipoles);
  dipoleMagnitudeTCL.get().get_double(dipoleMagnitude);
  seedRandTCL.set(to_string(seedRand+1));
  cerr << "seedRand="<<seedRand<<"\n";
  MusilRNG mr(seedRand);
  mr();
//  cerr << "rand="<<mr()<<"\n";
  DenseMatrix *m=scinew DenseMatrix(numDipoles, 6);
  for (int i=0; i<numDipoles; i++) {
    int elem = mr() * mesh->elems.size();
    cerr << "elem["<<i<<"]="<<elem<<"\n";
    Point p(mesh->elems[elem]->centroid());
    (*m)[i][0]=p.x();
    (*m)[i][1]=p.y();
    (*m)[i][2]=p.z();
    (*m)[i][3]=2*(mr()-0.5)*dipoleMagnitude;
    (*m)[i][4]=2*(mr()-0.5)*dipoleMagnitude;
    (*m)[i][5]=2*(mr()-0.5)*dipoleMagnitude;
  }
  omat->send(MatrixHandle(m));
}
} // End namespace SCIRun
