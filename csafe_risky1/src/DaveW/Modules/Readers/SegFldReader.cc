//static char *id="@(#) $Id$";

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

#include <DaveW/Datatypes/General/SegFldPort.h>
#include <DaveW/Datatypes/General/SegFld.h>
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

class SegFldReader : public Module {
    SegFldOPort* outport;
    TCLstring filename;
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
	    error("Error reading SegFld from file");
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
// Revision 1.2  2000/03/17 09:25:57  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.1  1999/09/01 07:19:52  dmw
// new DaveW modules
//
//
