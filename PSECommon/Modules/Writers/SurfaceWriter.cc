//static char *id="@(#) $Id$";

/*
 *  SurfaceWriter.cc: Surface Writer class
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
#include <SCICore/Persistent/Pstreams.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/SurfacePort.h>
#include <SCICore/CoreDatatypes/Surface.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class SurfaceWriter : public Module {
    SurfaceIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    SurfaceWriter(const clString& id);
    SurfaceWriter(const SurfaceWriter&, int deep=0);
    virtual ~SurfaceWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_SurfaceWriter(const clString& id) {
  return new SurfaceWriter(id);
}

SurfaceWriter::SurfaceWriter(const clString& id)
: Module("SurfaceWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew SurfaceIPort(this, "Input Data", SurfaceIPort::Atomic);
    add_iport(inport);
}

SurfaceWriter::SurfaceWriter(const SurfaceWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("SurfaceWriter::SurfaceWriter");
}

SurfaceWriter::~SurfaceWriter()
{
}

Module* SurfaceWriter::clone(int deep)
{
    return scinew SurfaceWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    SurfaceWriter* writer=(SurfaceWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void SurfaceWriter::execute()
{
    using SCICore::Containers::Pio;

    SurfaceHandle handle;
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
// Revision 1.2  1999/08/17 06:37:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:20  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:32  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:05  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 03:25:36  dav
// adding these files in too... should have been there already... oh well, sigh
//
//
