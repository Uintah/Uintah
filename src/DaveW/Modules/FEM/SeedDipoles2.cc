/*
 *  SeedDipoles2.cc:  From a mesh, seed some number of dipoles
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   October 2000
 *
 *  Copyright (C) 2000 SCI Group
 */
 
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MusilRNG.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <DaveW/share/share.h>

#include <iostream>
using std::cerr;

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::TclInterface;

class DaveWSHARE SeedDipoles2 : public Module {
  MeshIPort* imesh;
  MatrixOPort* omat;  
  TCLstring seedRandTCL;
  TCLstring numDipolesTCL;
  TCLstring dipoleMagnitudeTCL;
public:
  SeedDipoles2(const clString& id);
  virtual ~SeedDipoles2();
  virtual void execute();
};

extern "C" DaveWSHARE Module* make_SeedDipoles2(const clString& id) {
  return new SeedDipoles2(id);
}

SeedDipoles2::SeedDipoles2(const clString& id)
  : Module("SeedDipoles2", id, Filter), seedRandTCL("seedRandTCL", id, this),
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

SeedDipoles2::~SeedDipoles2()
{
}

void SeedDipoles2::execute()
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

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.1.2.1  2000/10/31 02:14:47  dmw
// merging DaveW HEAD changes into FIELD_BRANCH
//
// Revision 1.1  2000/10/29 03:51:45  dmw
// SeedDipoles will place dipoles randomly within a mesh
//
