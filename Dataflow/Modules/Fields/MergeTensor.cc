
/*
 *  Gradient.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Geometry/Point.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class MergeTensor : public Module {
    ScalarFieldIPort* infield;

    VectorFieldIPort* tensora;
    VectorFieldIPort* tensorb;

    VectorFieldOPort* outfield;
    VectorFieldOPort* outJ;
public:
    MergeTensor(const clString& id);
    virtual ~MergeTensor();
    virtual void execute();
};

extern "C" Module* make_MergeTensor(const clString& id) {
  return new MergeTensor(id);
}

MergeTensor::MergeTensor(const clString& id)
: Module("MergeTensor", id, Filter)
{
    infield=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(infield);

    tensora = new 
      VectorFieldIPort(this,"Tensor Diagonal",VectorFieldIPort::Atomic);
    add_iport(tensora);

    tensorb = new 
      VectorFieldIPort(this,"Tensor Off Diagonal",VectorFieldIPort::Atomic);
    add_iport(tensorb);

    // Create the output ports
    outfield=new VectorFieldOPort(this, "Vec Field", VectorFieldIPort::Atomic);
    add_oport(outfield);

    outJ=new VectorFieldOPort(this, "J", VectorFieldIPort::Atomic);
    add_oport(outJ);
}

MergeTensor::~MergeTensor()
{
}

void MergeTensor::execute()
{
  ScalarFieldHandle iff;
  if(!infield->get(iff))
    return;
  ScalarFieldRG* sfield=iff->getRG();
  if(!sfield){
    error("Gradient can't deal with this field");
    return;
  }

  // this is all that is needed to create vector field from potentials

  VectorFieldRG* vfield;

  vfield = new VectorFieldRG();

  vfield->resize(sfield->nx,sfield->ny,sfield->nz);
  Point bmin,bmax;
  sfield->get_bounds(bmin,bmax);
  vfield->set_bounds(bmin,bmax);

  int x;
  for(x=0;x<sfield->nx;x++)
    for(int y=0;y<sfield->ny;y++)
      for(int z=0;z<sfield->nz;z++) {
	vfield->grid(x,y,z) = sfield->gradient(x,y,z);
      }
  
  outfield->send(VectorFieldHandle(vfield));

  VectorFieldRG* jfield,*arg,*brg; // J = sigma*grad, where grad is vfield...

  VectorFieldHandle ta,tb;

  if (!tensora->get(ta))
    return;
  if (!tensorb->get(tb))
    return;

  arg = ta->getRG();
  brg = tb->getRG();

  if (!arg || !brg)
    return; // tensors aren't regular grids...

  jfield = new VectorFieldRG();

  jfield->resize(sfield->nx,sfield->ny,sfield->nz);
  sfield->get_bounds(bmin,bmax);
  jfield->set_bounds(bmin,bmax);
  
  // ok, now run through tensor and vectors, multiply and store...

  for(x=0;x<sfield->nx;x++)
    for(int y=0;y<sfield->ny;y++)
      for(int z=0;z<sfield->nz;z++) {
	Vector G = vfield->grid(x,y,z);
	Vector TD = arg->grid(x,y,z);
	Vector TOD = brg->grid(x,y,z);

	// now do matrix vector multiply...

	Vector nvec(TD.x()*G.x()+TOD.x()*G.y()+TOD.y()*G.z(),
		    TOD.x()*G.x() + TD.y()*G.y() + TOD.z()*G.z(),
		    TOD.y()*G.x() + TOD.z()*G.y() + TD.z()*G.z());

	jfield->grid(x,y,z) = nvec;
      }
  outJ->send(VectorFieldHandle(jfield));

}

} // End namespace SCIRun

