
/*
 *  ColumnMatrixWriter.cc: ColumnMatrix Writer class
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
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class ColumnMatrixWriter : public Module {
    ColumnMatrixIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    ColumnMatrixWriter(const clString& id);
    virtual ~ColumnMatrixWriter();
    virtual void execute();
};

extern "C" Module* make_ColumnMatrixWriter(const clString& id) {
  return new ColumnMatrixWriter(id);
}

ColumnMatrixWriter::ColumnMatrixWriter(const clString& id)
: Module("ColumnMatrixWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew ColumnMatrixIPort(this, "Input Data", ColumnMatrixIPort::Atomic);
    add_iport(inport);
}

ColumnMatrixWriter::~ColumnMatrixWriter()
{
}

void ColumnMatrixWriter::execute()
{

    ColumnMatrixHandle handle;
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
    Pio(*stream, handle);
    delete stream;
}

} // End namespace SCIRun

