
/*
 *  VectorFieldReader.cc: VectorField Reader class
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
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Datatypes/VectorField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class VectorFieldReader : public Module {
    VectorFieldOPort* outport;
    TCLstring filename;
    VectorFieldHandle handle;
    clString old_filename;
public:
    VectorFieldReader(const clString& id);
    virtual ~VectorFieldReader();
    virtual void execute();
};

extern "C" Module* make_VectorFieldReader(const clString& id) {
  return new VectorFieldReader(id);
}

VectorFieldReader::VectorFieldReader(const clString& id)
: Module("VectorFieldReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew VectorFieldOPort(this, "Output Data", VectorFieldIPort::Atomic);
    add_oport(outport);
}

VectorFieldReader::~VectorFieldReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    VectorFieldReader* reader=(VectorFieldReader*)cbdata;
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

void VectorFieldReader::execute()
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
	    error("Error reading VectorField from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}

} // End namespace SCIRun

