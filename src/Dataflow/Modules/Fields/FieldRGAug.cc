//static char *id="@(#) $Id$";

/*
 * FieldRGAug.cc: This file augments a regular grid with 
 *  a "points" file -> this can be used to visualize isosurfaces
 *  in the actual "space" of the heart.
 *
 * Peter-Pike Sloan - for Duke stuff
 */

#include <SCICore/Util/NotFinished.h>
#include <SCICore/Util/Timer.h>
#include <SCICore/Containers/BitArray1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/GeometryPort.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <PSECore/CommonDatatypes/MeshPort.h>
#include <SCICore/CoreDatatypes/Mesh.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGBase.h>
#include <SCICore/CoreDatatypes/ScalarFieldRG.h>
#include <SCICore/CoreDatatypes/ScalarFieldUG.h>
#include <SCICore/CoreDatatypes/VectorField.h>
#include <SCICore/CoreDatatypes/VectorFieldUG.h>
#include <SCICore/CoreDatatypes/VectorFieldRG.h>
#include <PSECore/CommonDatatypes/VectorFieldPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Multitask/ITC.h>
#include <SCICore/Multitask/Task.h>
#include <PSECore/Widgets/ScaledBoxWidget.h>

#include <stdlib.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class FieldRGAug : public Module {
  ScalarFieldIPort *ifield; // field to augment...
  ScalarFieldOPort *ofield; // field to augment...
  TCLstring filename;
  clString old_filename;
  ScalarFieldHandle isfh;

  ScalarFieldRGBase *isf; // pointer to the scalar field -> use handles...

  int nx,ny,nz;

  ScalarFieldRG *xf,*yf,*zf; // filtered to generate smaller mesh...

public:
  FieldRGAug(const clString& id);
  virtual ~FieldRGAug();
  virtual void execute();

  
};

Module* make_FieldRGAug(const clString& id) {
  return new FieldRGAug(id);
}


FieldRGAug::FieldRGAug(const clString& id)
: Module("FieldRGAug", id, Source), filename("filename", id, this)
{
  // Create the output data handle and port
  ifield=scinew ScalarFieldIPort(this, "Input Data", ScalarFieldIPort::Atomic);
  add_iport(ifield);

  ofield=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
  add_oport(ofield);
}

FieldRGAug::~FieldRGAug()
{
}

void FieldRGAug::execute()
{
  if (!ifield->get(isfh))
    return;

  isf = isfh->getRGBase();

  if (!isf)
    return; // must be some form of regular grid...

  if (!get_tcl_intvar(id,"nx",nx)) {
    error("Couldn't read var");
    return;
  }
  if (!get_tcl_intvar(id,"ny",ny)) {
    error("Couldn't read var");
    return;
  }
  if (!get_tcl_intvar(id,"nz",nz)) {
    error("Couldn't read var");
    return;
  }

  clString fn(filename.get());
  if(fn != old_filename){ // new file, read it in and stick it in...

    FILE *f = fopen(fn(),"r"); // try and read it...
    
    if (!f)
      return; // file sucks...
    
    LockArray3<Point> *new_array;

    new_array= new LockArray3<Point>;

    new_array->newsize(nx,ny,nz);
    
    for(int z=0;z<nz;z++)
      for(int y=0;y<ny;y++)
	for(int x=0;x<nx;x++) {
	  double newval[3];
	  if (3 != fscanf(f,"%lf %lf %lf",&newval[0],&newval[1],&newval[2])) {
	    error("Choked reading file!\n");
	    delete new_array;
	    return; // caput...
	  }
	  (*new_array)(x,y,z) = Point(newval[0],newval[1],newval[2]);
	}
    // array has been created, cycle through the scalar fields now...

    ScalarFieldRGBase *cur_sf=isf;

    while(cur_sf) {
      cur_sf->is_augmented = 1;
      cur_sf->aug_data = new_array;
      cur_sf = cur_sf->next; // next pointer...
    }
    ofield->send(isfh);
  }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.3  1999/08/18 20:19:40  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:27  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:42  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:10  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//
