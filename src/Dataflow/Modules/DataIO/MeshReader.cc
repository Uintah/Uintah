/*
 *  MeshReader.cc: Mesh Reader class
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
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class MeshReader : public Module {
    MeshOPort* outport;
    TCLstring filename;
    MeshHandle handle;
    clString old_filename;
public:
    MeshReader(const clString& id);
    virtual ~MeshReader();
    virtual void execute();
};

extern "C" Module* make_MeshReader(const clString& id) {
  return new MeshReader(id);
}

MeshReader::MeshReader(const clString& id)
: Module("MeshReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew MeshOPort(this, "Output Data", MeshIPort::Atomic);
    add_oport(outport);
}

MeshReader::~MeshReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    MeshReader* reader=(MeshReader*)cbdata;
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

void MeshReader::execute()
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
      error("Error reading Mesh from file");
      delete stream;
      return;
    }
    delete stream;
    int i;
    handle->locate2(Point(0,0,0), i, 0.00001);
    handle->compute_neighbors();
  }
  outport->send(handle);
}

} // End namespace SCIRun


