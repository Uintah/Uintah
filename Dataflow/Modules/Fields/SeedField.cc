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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
using std::cerr;

namespace SCIRun {

class SeedField : public Module {
  FieldIPort* imesh;
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
  imesh = scinew FieldIPort(this, "Mesh", FieldIPort::Atomic);
  add_iport(imesh);
  
  // Create the output ports
  omat=scinew MatrixOPort(this,"Dipole Seeds", MatrixIPort::Atomic);
  add_oport(omat);
}

SeedField::~SeedField()
{
}

// FIX_ME upgrade to new fields.
void SeedField::execute()
{
#if 0
  FieldHandle mesh;
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
#endif 
}
} // End namespace SCIRun
