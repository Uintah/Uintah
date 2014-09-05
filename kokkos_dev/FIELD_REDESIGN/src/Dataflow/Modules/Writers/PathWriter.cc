//static char *id="@(#) $Id$";

/*
 *  PathWriter.cc: Path Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Datatypes/Path.h>
#include <PSECore/Datatypes/PathPort.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>


namespace PSECommon {
namespace Modules {

using namespace SCICore::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class PathWriter : public Module {
    PathIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    PathWriter(const clString& id);
    virtual ~PathWriter();
    virtual void execute();
};

extern "C" Module* make_PathWriter(const clString& id) {
  return new PathWriter(id);
}

PathWriter::PathWriter(const clString& id)
: Module("PathWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew PathIPort(this, "Input Data", PathIPort::Atomic);
    add_iport(inport);
}

PathWriter::~PathWriter()
{
}

#if 0
static void watcher(double pd, void* cbdata)
{
    PathWriter* writer=(PathWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void PathWriter::execute()
{
    using SCICore::Containers::Pio;

    PathHandle handle;
    if(!inport->get(handle))
	return;
    clString fn(filename.get());
    
    if(fn == "")
	return;
    Piostream* stream;
    clString ft(filetype.get());

    if(ft=="Binary"){
	stream=scinew BinaryPiostream(fn, Piostream::Write);
    } else {
	stream=scinew TextPiostream(fn, Piostream::Write);
    }
    // Write the file
    //stream->watch_progress(watcher, (void*)this);
 
    Pio(*stream, handle);
    delete stream;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2.2.1  2000/09/28 03:15:34  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.2  2000/07/19 19:30:20  samsonov
// Moving from DaveW
//
// Revision 1.1  2000/07/18 23:14:12  samsonov
// PathWriter module is transfered from DaveW package
//
// Revision 1.2  2000/03/17 09:26:06  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.1  1999/12/02 22:06:50  dmw
// added reader/writer modules for camera path
//
//
