
/*
 *  DukeRawReader.cc: ScalarField Reader class
 *
 *  Written by:
 *   Reads raw files used by EEL lab at Duke
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

#include <stdio.h>

class DukeRawReader : public Module {
  ScalarFieldOPort* outport;
  TCLstring filename;
  ScalarFieldHandle handle;
  clString old_filename;

  int nx,ny,nz;       // dimensions
  double spx,spy,spz; // cell sizes
public:
  DukeRawReader(const clString& id);
  DukeRawReader(const DukeRawReader&, int deep=0);
  virtual ~DukeRawReader();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
  Module* make_DukeRawReader(const clString& id)
    {
      return scinew DukeRawReader(id);
    }
};

DukeRawReader::DukeRawReader(const clString& id)
: Module("DukeRawReader", id, Source), filename("filename", id, this)
{
  // Create the output data handle and port
  outport=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
  add_oport(outport);
}

DukeRawReader::DukeRawReader(const DukeRawReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
  NOT_FINISHED("DukeRawReader::DukeRawReader");
}

DukeRawReader::~DukeRawReader()
{
}

Module* DukeRawReader::clone(int deep)
{
  return scinew DukeRawReader(*this, deep);
}

void DukeRawReader::execute()
{
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

  if (!get_tcl_doublevar(id,"spx",spx)) {
    error("Couldn't read var");
    return;
  }
  if (!get_tcl_doublevar(id,"spy",spy)) {
    error("Couldn't read var");
    return;
  }
  if (!get_tcl_doublevar(id,"spz",spz)) {
    error("Couldn't read var");
    return;
  }

  clString fn(filename.get());
  if(!handle.get_rep() || fn != old_filename){

    // now loop through all of the files - 000 to number of frames...

    ScalarFieldRG*    last_ptr=0;
    ScalarFieldRG*    this_ptr=0;
    
    old_filename=fn;
    clString base_name(fn);

    clString work_name;

    char hun = '0',ten = '0', one = '1';
    int ndone=0;
    int finished=0;

    while (!finished) {
      work_name = base_name;
      work_name += hun;
      work_name += ten;
      work_name += one; // this is the file for this thing...

      
      FILE *f = fopen(work_name(),"r");

      if(!f){
	if (!ndone) {
	  cerr << "Error, couldn't open file!\n";
	  return; // Can't open file...
	}
	finished=1; // done with all of the scalar fields...
      } else {
	
	this_ptr = new ScalarFieldRG;
	this_ptr->resize(nx,ny,nz);
	Point pmin(0,0,0),pmax(nx*spx,ny*spy,nz*spz);
	this_ptr->set_bounds(pmin,pmax);

	this_ptr->compute_bounds();

	for(int z=0;z<nz;z++)
	  for(int y=0;y<ny;y++)
	    for(int x=0;x<nx;x++) {
	      double newval;
	      if (1 != fscanf(f,"%lf",&newval)) {
		error("Choked reading file!\n");
		delete this_ptr;
		return; // caput...
	      }
	      this_ptr->grid(x,y,z) = newval; // assign it...
	    }

	if (!ndone)
	  handle=(ScalarField*)this_ptr; // set the root

	if (last_ptr)
	  last_ptr->next = this_ptr;
	
	++ndone;

	cerr << "Did " << ndone << " " << work_name << "\n";

	last_ptr = this_ptr;
      }
      one = one + 1;
      if (one > '9') {
	ten = ten + 1;
	if (ten > '9') {
	  hun = hun+1; // shouldn't go over...
	  ten = '0';
	}
	one = '0';
      }
    }
  }
  outport->send(handle);
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template void Pio(Piostream&, ScalarFieldHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/LockingHandle.cc>

static void _dummy_(Piostream& p1, ScalarFieldHandle& p2)
{
  Pio(p1, p2);
}

#endif
#endif

