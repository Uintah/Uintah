//static char *id="@(#) $Id$";

/*
 *  ScalarFieldReader.cc: ScalarField Reader class
 *
 *  Written by:
 *   Steven G. Parker - hacked for multiple scalar fields by ppsloan...
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

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class MultiSFRGReader : public Module {
  ScalarFieldOPort* outport;
  TCLstring filename;
  ScalarFieldHandle handle;
  ScalarFieldHandle base;
  clString old_filename;
public:
  MultiSFRGReader(const clString& id);
  virtual ~MultiSFRGReader();
  virtual void execute();
};

extern "C" Module* make_MultiSFRGReader(const clString& id) {
  return new MultiSFRGReader(id);
}

MultiSFRGReader::MultiSFRGReader(const clString& id)
: Module("MultiSFRGReader", id, Source), filename("filename", id, this)
{
  // Create the output data handle and port
  outport=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
  add_oport(outport);
}

MultiSFRGReader::~MultiSFRGReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
  MultiSFRGReader* reader=(MultiSFRGReader*)cbdata;
  if(TCLTask::try_lock()){
    // Try the malloc lock once before we call update_progress
    // If we can't get it, then back off, since our caller might
    // have it locked
    if(!Task::test_malloc_lock()){
      TCLTask::unlock();
      return;
    }
    reader->update_progress(pd);
    TCLTask::unlock();
  }
}
#endif

void MultiSFRGReader::execute()
{
  using SCICore::Containers::Pio;

  clString fn(filename.get());
  if(!handle.get_rep() || fn != old_filename){

    // now loop through all of the files - 000 to number of frames...

    ScalarFieldHandle* crap; // for later...

    ScalarFieldRG*    last_ptr=0;

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

      Piostream* stream=auto_istream(work_name);
      
      if(!stream){
	if (!ndone) {
	  cerr << "Error, couldn't open file!\n";
	  return; // Can't open file...
	}
	finished=1; // done with all of the scalar fields...
      } else {
	
	// Read the file...
	Pio(*stream, base);
	crap = new ScalarFieldHandle;
	*crap = base; // add a fake reference...
	if (!ndone)
	  handle=base; // set the root
	if(!base.get_rep() || stream->error()){
	  error("Error reading ScalarField from file");
	  delete stream;
	  return;
	}
	delete stream;
	++ndone;

	cerr << "Did " << ndone << " " << work_name << "\n";

	if (last_ptr)
	  last_ptr->next = base.get_rep()->getRG();
	last_ptr = base.get_rep()->getRG();
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
// Revision 1.7  2000/03/17 09:27:12  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:06:54  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:55  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:51  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:49  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:35  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:48  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:26  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:53  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
