//static char *id="@(#) $Id$";

/*
 *  VectorFieldWriter.cc: VectorField Writer class
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
#include <CommonDatatypes/VectorFieldPort.h>
#include <CoreDatatypes/VectorField.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class VectorFieldWriter : public Module {
    VectorFieldIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    VectorFieldWriter(const clString& id);
    VectorFieldWriter(const VectorFieldWriter&, int deep=0);
    virtual ~VectorFieldWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_VectorFieldWriter(const clString& id) {
  return new VectorFieldWriter(id);
}

VectorFieldWriter::VectorFieldWriter(const clString& id)
: Module("VectorFieldWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew VectorFieldIPort(this, "Input Data", VectorFieldIPort::Atomic);
    add_iport(inport);
}

VectorFieldWriter::VectorFieldWriter(const VectorFieldWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("VectorFieldWriter::VectorFieldWriter");
}

VectorFieldWriter::~VectorFieldWriter()
{
}

Module* VectorFieldWriter::clone(int deep)
{
    return scinew VectorFieldWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    VectorFieldWriter* writer=(VectorFieldWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void VectorFieldWriter::execute()
{
    using SCICore::Containers::Pio;

    VectorFieldHandle handle;
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
// Revision 1.3  1999/07/07 21:10:33  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:06  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 03:25:36  dav
// adding these files in too... should have been there already... oh well, sigh
//
//
