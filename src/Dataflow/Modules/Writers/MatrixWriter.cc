//static char *id="@(#) $Id$";

/*
 *  MatrixWriter.cc: Matrix Writer class
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
#include <CommonDatatypes/MatrixPort.h>
#include <CoreDatatypes/Matrix.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class MatrixWriter : public Module {
    MatrixIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    MatrixWriter(const clString& id);
    MatrixWriter(const MatrixWriter&, int deep=0);
    virtual ~MatrixWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_MatrixWriter(const clString& id) {
  return new MatrixWriter(id);
}

MatrixWriter::MatrixWriter(const clString& id)
: Module("MatrixWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew MatrixIPort(this, "Input Data", MatrixIPort::Atomic);
    add_iport(inport);
}

MatrixWriter::MatrixWriter(const MatrixWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("MatrixWriter::MatrixWriter");
}

MatrixWriter::~MatrixWriter()
{
}

Module* MatrixWriter::clone(int deep)
{
    return scinew MatrixWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    MatrixWriter* writer=(MatrixWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void MatrixWriter::execute()
{
    using SCICore::Containers::Pio;

    MatrixHandle handle;
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
