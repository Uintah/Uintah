
/*
 *  ScalarFieldReader.cc: ScalarField Reader class
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
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class ScalarFieldReader : public Module {
    ScalarFieldOPort* outport;
    TCLstring filename;
    ScalarFieldHandle handle;
    clString old_filename;
public:
    ScalarFieldReader(const clString& id);
    virtual ~ScalarFieldReader();
    virtual void execute();
};

extern "C" Module* make_ScalarFieldReader(const clString& id) {
  return new ScalarFieldReader(id);
}

ScalarFieldReader::ScalarFieldReader(const clString& id)
: Module("ScalarFieldReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
    add_oport(outport);
}

ScalarFieldReader::~ScalarFieldReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    ScalarFieldReader* reader=(ScalarFieldReader*)cbdata;
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

void ScalarFieldReader::execute()
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
	    error("Error reading ScalarField from file");
	    delete stream;
	    return;
	}
	delete stream;
	handle->set_filename( fn );
    }
    outport->send(handle);
}

} // End namespace SCIRun

