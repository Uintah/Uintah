
/*
 *  ContourSetReader.cc: ContourSet Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/ContourSetPort.h>
#include <Packages/DaveW/Core/Datatypes/General/ContourSet.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class ContourSetReader : public Module {
    ContourSetOPort* outport;
    GuiString filename;
    ContourSetHandle handle;
    clString old_filename;
public:
    ContourSetReader(const clString& id);
    virtual ~ContourSetReader();
    virtual void execute();
};

extern "C" Module* make_ContourSetReader(const clString& id) {
  return new ContourSetReader(id);
}

ContourSetReader::ContourSetReader(const clString& id)
: Module("ContourSetReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew ContourSetOPort(this, "Output Data", ContourSetIPort::Atomic);
    add_oport(outport);
}

ContourSetReader::~ContourSetReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    ContourSetReader* reader=(ContourSetReader*)cbdata;
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

void ContourSetReader::execute()
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
	    error("Error reading ContourSet from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}
} // End namespace DaveW


