//static char *id="@(#) $Id$";

/*
 *  MeshWriter.cc: Mesh Writer class
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
#include <PSECore/CommonDatatypes/MeshPort.h>
#include <SCICore/CoreDatatypes/Mesh.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class MeshWriter : public Module {
    MeshIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    MeshWriter(const clString& id);
    MeshWriter(const MeshWriter&, int deep=0);
    virtual ~MeshWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_MeshWriter(const clString& id) {
  return new MeshWriter(id);
}

MeshWriter::MeshWriter(const clString& id)
: Module("MeshWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew MeshIPort(this, "Input Data", MeshIPort::Atomic);
    add_iport(inport);
}

MeshWriter::MeshWriter(const MeshWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("MeshWriter::MeshWriter");
}

MeshWriter::~MeshWriter()
{
}

Module* MeshWriter::clone(int deep)
{
    return scinew MeshWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    MeshWriter* writer=(MeshWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void MeshWriter::execute()
{
    using SCICore::Containers::Pio;

    MeshHandle handle;
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
// Revision 1.2  1999/04/27 22:58:04  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 03:25:35  dav
// adding these files in too... should have been there already... oh well, sigh
//
//
