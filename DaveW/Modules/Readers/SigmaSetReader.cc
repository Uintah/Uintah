//static char *id="@(#) $Id$";

/*
 *  SigmaSetReader.cc: SigmaSet Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <DaveW/Datatypes/General/SigmaSetPort.h>
#include <DaveW/Datatypes/General/SigmaSet.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class SigmaSetReader : public Module {
    SigmaSetOPort* outport;
    TCLstring filename;
    SigmaSetHandle handle;
    clString old_filename;
public:
    SigmaSetReader(const clString& id);
    virtual ~SigmaSetReader();
    virtual void execute();
};

Module* make_SigmaSetReader(const clString& id) {
  return new SigmaSetReader(id);
}

SigmaSetReader::SigmaSetReader(const clString& id)
: Module("SigmaSetReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew SigmaSetOPort(this, "Output Data", SigmaSetIPort::Atomic);
    add_oport(outport);
}

SigmaSetReader::~SigmaSetReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    SigmaSetReader* reader=(SigmaSetReader*)cbdata;
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

void SigmaSetReader::execute()
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
	    error("Error reading SigmaSet from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/01 07:19:52  dmw
// new DaveW modules
//
//
