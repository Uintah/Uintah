
/*
 *  ColumnMatrixReader.cc: ColumnMatrix Reader class
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
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {



class ColumnMatrixReader : public Module {
    ColumnMatrixOPort* outport;
    TCLstring filename;
    ColumnMatrixHandle handle;
    clString old_filename;
public:
    ColumnMatrixReader(const clString& id);
    virtual ~ColumnMatrixReader();
    virtual void execute();
};

extern "C" Module* make_ColumnMatrixReader(const clString& id) {
  return new ColumnMatrixReader(id);
}

ColumnMatrixReader::ColumnMatrixReader(const clString& id)
: Module("ColumnMatrixReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew ColumnMatrixOPort(this, "Output Data", ColumnMatrixIPort::Atomic);
    add_oport(outport);
}

ColumnMatrixReader::~ColumnMatrixReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    ColumnMatrixReader* reader=(ColumnMatrixReader*)cbdata;
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

void ColumnMatrixReader::execute()
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
	    error("Error reading ColumnMatrix from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}

} // End namespace SCIRun

