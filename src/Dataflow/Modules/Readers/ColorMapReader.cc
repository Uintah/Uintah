//static char *id="@(#) $Id$";

/*
 *  ColorMapReader.cc: ColorMap Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/ColorMapPort.h>
#include <SCICore/CoreDatatypes/ColorMap.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class ColorMapReader : public Module {
    ColorMapOPort* outport;
    TCLstring filename;
    ColorMapHandle handle;
    clString old_filename;
public:
    ColorMapReader(const clString& id);
    virtual ~ColorMapReader();
    virtual void execute();
};

Module* make_ColorMapReader(const clString& id) {
  return new ColorMapReader(id);
}

ColorMapReader::ColorMapReader(const clString& id)
: Module("ColorMapReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew ColorMapOPort(this, "Output Data", ColorMapIPort::Atomic);
    add_oport(outport);
}

ColorMapReader::~ColorMapReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    ColorMapReader* reader=(ColorMapReader*)cbdata;
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

void ColorMapReader::execute()
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
	    error("Error reading ColorMap from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  1999/08/19 23:17:49  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:47  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:33  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:47  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:24  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:51  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 02:26:25  dav
// Sorry for all this mail traffic Steve... :'(
//
//
