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

#include <DaveW/Datatypes/General/Path.h>
#include <DaveW/Datatypes/General/PathPort.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
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

Module* make_PathWriter(const clString& id) {
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
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/12/02 22:06:50  dmw
// added reader/writer modules for camera path
//
//
