//static char *id="@(#) $Id$";

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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class DukeRawReader : public Module {
  ScalarFieldOPort* outport;
  TCLstring filename;
  ScalarFieldHandle handle;
  clString old_filename;

  int nx,ny,nz;       // dimensions
  double spx,spy,spz; // cell sizes
public:
  DukeRawReader(const clString& id);
  virtual ~DukeRawReader();
  virtual void execute();
};

Module* make_DukeRawReader(const clString& id) {
  return new DukeRawReader(id);
}

DukeRawReader::DukeRawReader(const clString& id)
: Module("DukeRawReader", id, Source), filename("filename", id, this)
{
  // Create the output data handle and port
  outport=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
  add_oport(outport);
}

DukeRawReader::~DukeRawReader()
{
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/10/07 02:06:54  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:53  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:50  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:47  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:33  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:47  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:52  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
