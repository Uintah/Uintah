/*
 * FieldRGAug.cc: This file augments a regular grid with 
 *  a "points" file -> this can be used to visualize isosurfaces
 *  in the actual "space" of the heart.
 *
 * Peter-Pike Sloan - for Duke stuff
 */

#include <Classlib/NotFinished.h>
#include <Classlib/Timer.h>
#include <Classlib/BitArray1.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geom/Pt.h>
#include <Geom/Material.h>
#include <Geom/Tri.h>
#include <Geom/Triangles.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/VectorField.h>
#include <Datatypes/VectorFieldUG.h>
#include <Datatypes/VectorFieldRG.h>
#include <Datatypes/VectorFieldPort.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <Multitask/ITC.h>
#include <Multitask/Task.h>
#include <Widgets/ScaledBoxWidget.h>

#include <stdlib.h>
#include <stdio.h>

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

extern "C" {
  Module* make_FieldRGAug(const clString& id)
    {
      return scinew FieldRGAug(id);
    }
};


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

