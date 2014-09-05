//static char *id="@(#) $Id$";

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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  2000/03/17 09:26:59  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/08/25 03:47:48  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:45  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:41  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:29  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:43  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:12  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//
