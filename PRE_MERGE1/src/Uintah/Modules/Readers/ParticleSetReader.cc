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

#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/Particles/ParticleSetPort.h>
#include <Datatypes/Particles/ParticleSet.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLTask.h>
#include <TclInterface/TCLvar.h>

namespace Uintah {
namespace Modules {

using namespace Uintah::Datatypes;
using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class ParticleSetReader : public Module {
    ParticleSetOPort* outport;
    TCLstring filename;
    ParticleSetHandle handle;
    clString old_filename;
public:
    ParticleSetReader(const clString& id);
    ParticleSetReader(const ParticleSetReader&, int deep=0);
    virtual ~ParticleSetReader();
    virtual Module* clone(int deep);
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

ParticleSetReader::ParticleSetReader(const ParticleSetReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("ParticleSetReader::ParticleSetReader");
}

ParticleSetReader::~ParticleSetReader()
{
}

Module* ParticleSetReader::clone(int deep)
{
    return scinew ParticleSetReader(*this, deep);
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
} // End namespace PSECommon

//
// $Log$
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
