//static char *id="@(#) $Id$";

/*
 *  GeomReader.cc: Geom Reader class
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
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Geom/GeomScene.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::PersistentSpace;

class GeomReader : public Module {
    GeometryOPort* outport;
    TCLstring filename;
    clString old_filename;
public:
    GeomReader(const clString& id);
    virtual ~GeomReader();
    virtual void execute();
};

Module* make_GeomReader(const clString& id) {
  return new GeomReader(id);
}

GeomReader::GeomReader(const clString& id)
: Module("GeomReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew GeometryOPort(this, "Output_Data", GeometryIPort::Atomic);
    add_oport(outport);
}

GeomReader::~GeomReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    GeomReader* reader=(GeomReader*)cbdata;
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

void GeomReader::execute()
{
    clString fn(filename.get());
    if(fn != old_filename){
	outport->delAll();
	old_filename=fn;
	Piostream* stream=auto_istream(fn);
	if(!stream){
	    error(clString("Error reading file: ")+filename.get());
	    return; // Can't open file...
	}
	// Read the file...
	GeomScene scene;
	Pio(*stream, scene);
	if(!scene.top || stream->error()){
	    error("Error reading Geom from file");
	    delete stream;
	    return;
	}
	delete stream;
	outport->addObj(scene.top, old_filename);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.5  1999/08/25 03:47:54  sparker
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
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
