
/*
 *  TYPEReader.cc: TYPE Reader class
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
#include <Dataflow/ModuleList.h>
#include <Datatypes/TYPEPort.h>
#include <Datatypes/TYPE.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

class TYPEReader : public Module {
    TYPEOPort* outport;
    TCLstring filename;
    TYPEHandle handle;
    clString old_filename;
public:
    TYPEReader(const clString& id);
    TYPEReader(const TYPEReader&, int deep=0);
    virtual ~TYPEReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_TYPEReader(const clString& id)
{
    return new TYPEReader(id);
}

#include "TYPERegister.h"

TYPEReader::TYPEReader(const clString& id)
: Module("TYPEReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=new TYPEOPort(this, "Output Data", TYPEIPort::Atomic);
    add_oport(outport);
}

TYPEReader::TYPEReader(const TYPEReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("TYPEReader::TYPEReader");
}

TYPEReader::~TYPEReader()
{
}

Module* TYPEReader::clone(int deep)
{
    return new TYPEReader(*this, deep);
}

static void watcher(double pd, void* cbdata)
{
    TYPEReader* reader=(TYPEReader*)cbdata;
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

void TYPEReader::execute()
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
	stream->watch_progress(watcher, (void*)this);
	Pio(*stream, handle);
	cerr << "Pio done!\n";
	if(!handle.get_rep()){
	    error("Error reading TYPE from file");
	    delete stream;
	    return;
	}
	cerr << "deleting stream:" << stream << "\n";
	delete stream;
	cerr << "delete done\n";
    }
    outport->send(handle);
}
