//static char *id="@(#) $Id$";

/*
 *  MatrixReader.cc: Matrix Reader class
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
#include <PSECore/CommonDatatypes/MatrixPort.h>
#include <SCICore/CoreDatatypes/Matrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class MatrixReader : public Module {
    MatrixOPort* outport;
    TCLstring filename;
    MatrixHandle handle;
    clString old_filename;
public:
    MatrixReader(const clString& id);
    MatrixReader(const MatrixReader&, int deep=0);
    virtual ~MatrixReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_MatrixReader(const clString& id) {
  return new MatrixReader(id);
}

MatrixReader::MatrixReader(const clString& id)
: Module("MatrixReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew MatrixOPort(this, "Output Data", MatrixIPort::Atomic);
    add_oport(outport);
}

MatrixReader::MatrixReader(const MatrixReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("MatrixReader::MatrixReader");
}

MatrixReader::~MatrixReader()
{
}

Module* MatrixReader::clone(int deep)
{
    return scinew MatrixReader(*this, deep);
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    MatrixReader* reader=(MatrixReader*)cbdata;
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

void MatrixReader::execute()
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
	    error("Error reading Matrix from file");
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
// Revision 1.2  1999/08/17 06:37:34  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:48  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:25  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:52  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 02:38:09  dav
// more things that should have been there but were not
//
//
