//static char *id="@(#) $Id$";

/*
 *  SegFldWriter.cc: SegFld Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <DaveW/Datatypes/General/SegFld.h>
#include <DaveW/Datatypes/General/SegFldPort.h>
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

class SegFldWriter : public Module {
    SegFldIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    SegFldWriter(const clString& id);
    virtual ~SegFldWriter();
    virtual void execute();
};

extern "C" Module* make_SegFldWriter(const clString& id) {
  return new SegFldWriter(id);
}

SegFldWriter::SegFldWriter(const clString& id)
: Module("SegFldWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew SegFldIPort(this, "Input Data", SegFldIPort::Atomic);
    add_iport(inport);
}

SegFldWriter::~SegFldWriter()
{
}

#if 0
static void watcher(double pd, void* cbdata)
{
    SegFldWriter* writer=(SegFldWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void SegFldWriter::execute()
{
    using SCICore::Containers::Pio;

    SegFldHandle handle;
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
// Revision 1.2  2000/03/17 09:26:06  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.1  1999/09/01 07:21:01  dmw
// new DaveW modules
//
//
