//static char *id="@(#) $Id$";

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

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/GeometryPort.h>
#include <SCICore/Geom/GeomScene.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::PersistentSpace;

class GeomReader : public Module {
    GeometryOPort* outport;
    TCLstring filename;
    clString old_filename;
public:
    GeomReader(const clString& id);
    GeomReader(const GeomReader&, int deep=0);
    virtual ~GeomReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_GeomReader(const clString& id) {
  return new GeomReader(id);
}

GeomReader::GeomReader(const clString& id)
: Module("GeomReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew GeometryOPort(this, "Output_Data", GeometryIPort::Atomic);
    add_oport(outport);
}

GeomReader::GeomReader(const GeomReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("GeomReader::GeomReader");
}

GeomReader::~GeomReader()
{
}

Module* GeomReader::clone(int deep)
{
    return scinew GeomReader(*this, deep);
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:33  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:47  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
