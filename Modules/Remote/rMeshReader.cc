
/*
 *  MeshReader.cc: Mesh Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *  Hacked by:
 *   Michelle Miller for remote testing without UI
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <stdio.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

using sci::MeshHandle;

class rMeshReader : public Module {
    MeshOPort* outport;
    TCLstring filename;
    MeshHandle handle;
    clString old_filename;
public:
    rMeshReader(const clString& id);
    rMeshReader(const rMeshReader&, int deep=0);
    virtual ~rMeshReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_rMeshReader(const clString& id)
{
    return scinew rMeshReader(id);
}
};

rMeshReader::rMeshReader(const clString& id)
: Module("rMeshReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew MeshOPort(this, "Output Data", MeshIPort::Atomic);
    add_oport(outport);
}

rMeshReader::rMeshReader(const rMeshReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("rMeshReader::rMeshReader");
}

rMeshReader::~rMeshReader()
{
}

Module* rMeshReader::clone(int deep)
{
    return scinew rMeshReader(*this, deep);
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    rMeshReader* reader=(rMeshReader*)cbdata;
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

void rMeshReader::execute()
{
printf ("Entering rMeshReader::execute()\n");
    clString fn(filename.get());
    //clString fn = "/home/sci/data1/development/data/small.mesh";
printf ("setup the readfile using auto_istream() rMeshReader::execute()\n");
    if(!handle.get_rep() || fn != old_filename){
	old_filename=fn;
	Piostream* stream=auto_istream(fn);
	if(!stream){
	    error(clString("Error reading file: ")+filename.get());
	    return; // Can't open file...
	}
	// Read the file...
	//stream->watch_progress(watcher, (void*)this);
printf ("read file into meshHandle - rMeshReader::execute()\n");
	Pio(*stream, handle);
	if(!handle.get_rep() || stream->error()){
	    error("Error reading Mesh from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
printf ("send meshHandle out oport - rMeshReader::execute()\n");
    outport->send(handle);
}
