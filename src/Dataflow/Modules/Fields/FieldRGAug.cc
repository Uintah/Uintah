
/*
 * FieldRGAug.cc: This file augments a regular grid with 
 *  a "points" file -> this can be used to visualize isosurfaces
 *  in the actual "space" of the heart.
 *
 * Peter-Pike Sloan - for Duke stuff
 */

#include <Core/Util/Timer.h>
#include <Core/Containers/BitArray1.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomTriangles.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRGBase.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Core/Datatypes/VectorField.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>
#include <Dataflow/Widgets/ScaledBoxWidget.h>

#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {


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

extern "C" Module* make_FieldRGAug(const clString& id) {
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

} // End namespace SCIRun

