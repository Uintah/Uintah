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

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class MatrixWriter : public Module {
    MatrixIPort* inport;
    TCLstring filename;
    TCLstring filetype;
    TCLint split;
public:
    MatrixWriter(const clString& id);
    virtual ~MatrixWriter();
    virtual void execute();
};

extern "C" Module* make_MatrixWriter(const clString& id) {
  return new MatrixWriter(id);
}

MatrixWriter::MatrixWriter(const clString& id)
: Module("MatrixWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this), split("split", id, this)
{
    // Create the output data handle and port
    inport=scinew MatrixIPort(this, "Input Data", MatrixIPort::Atomic);
    add_iport(inport);
}

MatrixWriter::~MatrixWriter()
{
}

void MatrixWriter::execute()
{

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

    handle->set_raw( split.get() );

    Pio(*stream, handle);
    delete stream;
}

} // End namespace SCIRun

