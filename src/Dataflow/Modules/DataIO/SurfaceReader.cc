
/*
 *  SurfaceReader.cc: Surface Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/Surface.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class SurfaceReader : public Module {
    SurfaceOPort* outport;
    TCLstring filename;
    SurfaceHandle handle;
    clString old_filename;
public:
    SurfaceReader(const clString& id);
    virtual ~SurfaceReader();
    virtual void execute();
};

extern "C" Module* make_SurfaceReader(const clString& id) {
  return new SurfaceReader(id);
}

SurfaceReader::SurfaceReader(const clString& id)
: Module("SurfaceReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew SurfaceOPort(this, "Output Data", SurfaceIPort::Atomic);
    add_oport(outport);
}

SurfaceReader::~SurfaceReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    SurfaceReader* reader=(SurfaceReader*)cbdata;
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

void SurfaceReader::execute()
{

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
	    error("Error reading Surface from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}

} // End namespace SCIRun


