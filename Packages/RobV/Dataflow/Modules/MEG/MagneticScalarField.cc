/*
 *  MagneticScalarField.cc:  Unfinished modules
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   August 1999
 *
 *  Copyright (C) 1999 SCI Group
 */


#include <Packages/RobV/Core/Datatypes/MEG/VectorFieldMI.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <iostream>
using std::cerr;

namespace RobV {
using namespace SCIRun;

class MagneticScalarField : public Module {
  VectorFieldHandle vfh;
  VectorFieldIPort* magneticFieldP;
  SurfaceIPort* isurfaceP;
  SurfaceOPort* magnitudeFieldP;
public:   
  MagneticScalarField(const clString& id);
  virtual ~MagneticScalarField();
  virtual void execute();
  void parallel(int proc);

  int np;
  int nelems;
  TriSurface* magField;
  Mutex mutex;
  MeshHandle mesh;
  VectorFieldHandle mField;
};


extern "C" Module* make_MagneticScalarField(const clString& id)
{
    return new MagneticScalarField(id);
}

MagneticScalarField::MagneticScalarField(const clString& id): Module("MagneticScalarField", id, Filter), mutex("MagneticScalarField lock for adding bc's")
{
  magneticFieldP=new VectorFieldIPort(this, "MagneticField", VectorFieldIPort::Atomic);
  add_iport(magneticFieldP);

  isurfaceP = new SurfaceIPort(this, "ISurface", SurfaceIPort::Atomic);
  add_iport(isurfaceP);

  // Create the output port
  magnitudeFieldP=new SurfaceOPort(this, "magnitudeField", SurfaceIPort::Atomic);
  add_oport(magnitudeFieldP);
}

MagneticScalarField::~MagneticScalarField()
{
}

void MagneticScalarField::parallel(int proc)
{

  int su=proc*nelems/np;
  int eu=(proc+1)*nelems/np;

  for (int i=su; i<eu; i++) {
    
    Element* e = mesh->elems[i];

    Vector magV;
    Point centroid = e->centroid();
    int ii = i;
    ((VectorFieldMI*)(mField.get_rep()))->interpolate(centroid, magV,ii);  

    double l = magV.length();

    mutex.lock();
    magField->bcIdx.add(i);  //correct use?
    magField->bcVal.add(l);
    mutex.unlock();
  }
}

void MagneticScalarField::execute() {

  if (!magneticFieldP->get(mField)) return;

  SurfaceHandle isurf;
  if(!isurfaceP->get(isurf))
    return;
  TriSurface *ts=isurf->getTriSurface();

  ////  //copy over input surface  //correct?

  magField = new TriSurface();

  //  Array1<Vector> normals;
  //isurf->get_surfnodes(nodes);
  //Array1<Point> pts(nodes.size());

  *magField=*ts;

  //  for (i=0; ts && i<isurf->normals.size(); i++) 
  //  normals.add((transformPt(m, ts->normals[i].point(), Vector(0,0,0), 1,1,1)).vector());
//if (normals.size()) magField->normals=normals;
//magField->points=pts;
  //////

  mesh = ((VectorFieldMI*)mField.get_rep())->getMesh();

  if (mesh.get_rep() == NULL) {
    cerr << "Error getting mesh\n";
    exit(0);
  }

  nelems = mesh->elems.size(); 

  VectorFieldMI* mi = dynamic_cast<VectorFieldMI*> (mField.get_rep());
  
  //parallel
  if (mi != NULL) {   //parallel for only vfmi
    np=Thread::numProcessors();
    if (np>4) np/=2;	// using half the processors.
  } else np = 1;
 
  Thread::parallel(Parallel<MagneticScalarField>(this, &MagneticScalarField::parallel), np, true);

//  Task::multiprocess(np, do_parallel, this);
  
  magnitudeFieldP->send(magField);

} // End namespace RobV
}

