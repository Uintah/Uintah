//static char *id="@(#) $Id$";

/*
 *  TYPEWriter.cc: TYPE Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Util/NotFinished.h>
#include <Persistent/Pstreams.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/TYPEPort.h>
#include <CommonDatatypes/TYPE.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class TYPEWriter : public Module {
    TYPEIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    TYPEWriter(const clString& id);
    TYPEWriter(const TYPEWriter&, int deep=0);
    virtual ~TYPEWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_TYPEWriter(const clString& id) {
  return new TYPEWriter(id);
}

TYPEWriter::TYPEWriter(const clString& id)
: Module("TYPEWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew TYPEIPort(this, "Input Data", TYPEIPort::Atomic);
    add_iport(inport);
}

TYPEWriter::TYPEWriter(const TYPEWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("TYPEWriter::TYPEWriter");
}

TYPEWriter::~TYPEWriter()
{
}

Module* TYPEWriter::clone(int deep)
{
    return scinew TYPEWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    TYPEWriter* writer=(TYPEWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void TYPEWriter::execute()
{
    TYPEHandle handle;
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
// Revision 1.1  1999/07/27 16:58:21  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
