
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/SigmaSetPort.h>
#include <Datatypes/SigmaSet.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

class SigmaSetReader : public Module {
    SigmaSetOPort* outport;
    TCLstring filename;
    SigmaSetHandle handle;
    clString old_filename;
public:
    SigmaSetReader(const clString& id);
    SigmaSetReader(const SigmaSetReader&, int deep=0);
    virtual ~SigmaSetReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SigmaSetReader(const clString& id)
{
    return scinew SigmaSetReader(id);
}
}

SigmaSetReader::SigmaSetReader(const clString& id)
: Module("SigmaSetReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew SigmaSetOPort(this, "Output Data", SigmaSetIPort::Atomic);
    add_oport(outport);
}

SigmaSetReader::SigmaSetReader(const SigmaSetReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("SigmaSetReader::SigmaSetReader");
}

SigmaSetReader::~SigmaSetReader()
{
}

Module* SigmaSetReader::clone(int deep)
{
    return scinew SigmaSetReader(*this, deep);
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

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template void Pio(Piostream&, SigmaSetHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/LockingHandle.cc>

static void _dummy_(Piostream& p1, SigmaSetHandle& p2)
{
    Pio(p1, p2);
}

#endif
#endif

