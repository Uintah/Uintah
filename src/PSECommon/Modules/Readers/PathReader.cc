//static char *id="@(#) $Id$";

/*
 *  PathReader.cc: Path Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Datatypes/PathPort.h>
#include <SCICore/Datatypes/Path.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace SCICore::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class PathReader : public Module {
    PathOPort* outport;
    TCLstring filename;
    PathHandle handle;
    clString old_filename;
public:
    PathReader(const clString& id);
    virtual ~PathReader();
    virtual void execute();
};

extern "C" Module* make_PathReader(const clString& id) {
  return new PathReader(id);
}

PathReader::PathReader(const clString& id)
: Module("PathReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew PathOPort(this, "Output Data", PathIPort::Atomic);
    add_oport(outport);
}

PathReader::~PathReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    PathReader* reader=(PathReader*)cbdata;
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

void PathReader::execute()
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
	    error("Error reading Path from file");
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
// Revision 1.2.2.3  2000/11/01 23:02:57  mcole
// Fix for previous merge from trunk
//
// Revision 1.2.2.1  2000/09/28 03:16:04  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.2  2000/07/19 19:29:35  samsonov
// Moving from DaveW
//
// Revision 1.1  2000/07/18 23:12:56  samsonov
// Added PathReader module
//
// Revision 1.2  2000/03/17 09:25:57  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.1  1999/12/02 22:06:46  dmw
// added reader/writer modules for camera path
//
//
