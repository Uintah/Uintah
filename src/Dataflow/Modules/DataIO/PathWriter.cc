
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

#include <Core/Datatypes/Path.h>
#include <Dataflow/Ports/PathPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>


namespace SCIRun {


class PathWriter : public Module {
    PathIPort* inport;
    GuiString filename;
    GuiString filetype;
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

} // End namespace SCIRun

