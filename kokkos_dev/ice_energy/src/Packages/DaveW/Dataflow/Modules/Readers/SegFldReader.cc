
/*
 *  SegFldReader.cc: SegFld Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/SegFldPort.h>
#include <Packages/DaveW/Core/Datatypes/General/SegFld.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class SegFldReader : public Module {
    SegFldOPort* outport;
    GuiString filename;
    SegFldHandle handle;
    clString old_filename;
public:
    SegFldReader(const clString& id);
    virtual ~SegFldReader();
    virtual void execute();
};

extern "C" Module* make_SegFldReader(const clString& id) {
  return new SegFldReader(id);
}

SegFldReader::SegFldReader(const clString& id)
: Module("SegFldReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew SegFldOPort(this, "Output Data", SegFldIPort::Atomic);
    add_oport(outport);
}

SegFldReader::~SegFldReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    SegFldReader* reader=(SegFldReader*)cbdata;
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

void SegFldReader::execute()
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
	    error("Error reading SegFld from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}
} // End namespace DaveW


