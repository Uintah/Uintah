//static char *id="@(#) $Id$";

/*
 * FieldRGAug.cc: This file augments a regular grid with 
 *  a "points" file -> this can be used to visualize isosurfaces
 *  in the actual "space" of the heart.
 *
 * Peter-Pike Sloan - for Duke stuff
 */

#include <Util/NotFinished.h>
#include <Util/Timer.h>
#include <Containers/BitArray1.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/GeomObj.h>
#include <Geom/GeomGroup.h>
#include <Geom/GeomLine.h>
#include <Geom/Pt.h>
#include <Geom/Material.h>
#include <Geom/GeomTri.h>
#include <Geom/GeomTriangles.h>
#include <CommonDatatypes/MeshPort.h>
#include <CoreDatatypes/Mesh.h>
#include <CommonDatatypes/ScalarFieldPort.h>
#include <CoreDatatypes/ScalarFieldRGBase.h>
#include <CoreDatatypes/ScalarFieldRG.h>
#include <CoreDatatypes/ScalarFieldUG.h>
#include <CoreDatatypes/VectorField.h>
#include <CoreDatatypes/VectorFieldUG.h>
#include <CoreDatatypes/VectorFieldRG.h>
#include <CommonDatatypes/VectorFieldPort.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLvar.h>
#include <Multitask/ITC.h>
#include <Multitask/Task.h>
#include <Widgets/ScaledBoxWidget.h>

#include <stdlib.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
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
  FieldRGAug(const FieldRGAug&, int deep);
  virtual ~FieldRGAug();
  virtual Module* clone(int deep);
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

FieldRGAug::FieldRGAug(const FieldRGAug& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
  NOT_FINISHED("FieldRGAug::FieldRGAug");
}

FieldRGAug::~FieldRGAug()
{
}

Module* FieldRGAug::clone(int deep)
{
  return scinew FieldRGAug(*this, deep);
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
