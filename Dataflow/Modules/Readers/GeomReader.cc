
/*
 *  GeomReader.cc: Geom Reader class
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
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/GeomScene.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class GeomReader : public Module {
    GeometryOPort* outport;
    TCLstring filename;
    clString old_filename;
public:
    GeomReader(const clString& id);
    virtual ~GeomReader();
    virtual void execute();
};

extern "C" Module* make_GeomReader(const clString& id) {
  return new GeomReader(id);
}

GeomReader::GeomReader(const clString& id)
: Module("GeomReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew GeometryOPort(this, "Output_Data", GeometryIPort::Atomic);
    add_oport(outport);
}

GeomReader::~GeomReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    GeomReader* reader=(GeomReader*)cbdata;
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

void GeomReader::execute()
{
    clString fn(filename.get());
    if(fn != old_filename){
	outport->delAll();
	old_filename=fn;
	Piostream* stream=auto_istream(fn);
	if(!stream){
	    error(clString("Error reading file: ")+filename.get());
	    return; // Can't open file...
	}
	// Read the file...
	GeomScene scene;
	Pio(*stream, scene);
	if(!scene.top || stream->error()){
	    error("Error reading Geom from file");
	    delete stream;
	    return;
	}
	delete stream;
	outport->addObj(scene.top, old_filename);
    }
}

} // End namespace SCIRun

