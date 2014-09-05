//static char *id="@(#) $Id$";

/*
 *  MatrixReader.cc: Matrix Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/Matrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class MatrixReader : public Module {
    MatrixOPort* outport;
    TCLstring filename;
    MatrixHandle handle;
    clString old_filename;
public:
    MatrixReader(const clString& id);
    virtual ~MatrixReader();
    virtual void execute();
};

extern "C" Module* make_MatrixReader(const clString& id) {
  return new MatrixReader(id);
}

MatrixReader::MatrixReader(const clString& id)
: Module("MatrixReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew MatrixOPort(this, "Output Data", MatrixIPort::Atomic);
    add_oport(outport);
}

MatrixReader::~MatrixReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    MatrixReader* reader=(MatrixReader*)cbdata;
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

void MatrixReader::execute()
{
    using SCICore::Containers::Pio;

    clString fn(filename.get());
    if(!handle.get_rep() || fn != old_filename){
	old_filename=fn;
	Piostream* stream=auto_istream(fn);
	if(!stream){
	    error(clString("Error reading file: ")+filename.get());
	    return; // Can't open file...
	}
	// Read the file...
//	stream->watch_progress(watcher, (void*)this);
	Pio(*stream, handle);
	if(!handle.get_rep() || stream->error()){
	    error("Error reading Matrix from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  2000/03/17 09:27:12  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/08/25 03:47:54  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:50  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:48  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:34  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:48  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:25  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:52  dav
// updates in Modules for Datatypes
//
// Revision 1.1  1999/04/25 02:38:09  dav
// more things that should have been there but were not
//
//
