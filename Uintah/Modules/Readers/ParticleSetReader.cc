//static char *id="@(#) $Id$";

/*
 *  ParticleSetReader.cc: ParticleSet Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECore/Dataflow/Module.h>

#include <Uintah/Datatypes/Particles/ParticleSetPort.h>
#include <Uintah/Datatypes/Particles/ParticleSet.h>

namespace Uintah {
namespace Modules {

using namespace Uintah::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class ParticleSetReader : public Module {
    ParticleSetOPort* outport;
    TCLstring filename;
    ParticleSetHandle handle;
    clString old_filename;
public:
    ParticleSetReader(const clString& id);
    virtual ~ParticleSetReader();
    virtual void execute();
};

Module* make_ParticleSetReader(const clString& id) {
  return new ParticleSetReader(id);
}

ParticleSetReader::ParticleSetReader(const clString& id)
: Module("ParticleSetReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew ParticleSetOPort(this, "Output Data", ParticleSetIPort::Atomic);
    add_oport(outport);
}

ParticleSetReader::~ParticleSetReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    ParticleSetReader* reader=(ParticleSetReader*)cbdata;
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

void ParticleSetReader::execute()
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
	    error("Error reading ParticleSet from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}

} // End namespace Modules
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/08/19 23:18:09  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:24  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:40:13  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 17:08:59  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:11:10  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//
