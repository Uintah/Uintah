
/*
 *  ScalarFieldWriter.cc: ScalarField Writer class
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
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class ScalarFieldWriter : public Module {
  ScalarFieldIPort* inport;
  TCLstring filename;
  TCLstring filetype;
  TCLint split;
public:
  ScalarFieldWriter(const clString& id);
  virtual ~ScalarFieldWriter();
  virtual void execute();
};

extern "C" Module* make_ScalarFieldWriter(const clString& id) {
  return new ScalarFieldWriter(id);
}

ScalarFieldWriter::ScalarFieldWriter(const clString& id)
: Module("ScalarFieldWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this),
  split("split", id, this)
{
    // Create the output data handle and port
    inport=scinew ScalarFieldIPort(this, "Input Data", ScalarFieldIPort::Atomic);
    add_iport(inport);
}

ScalarFieldWriter::~ScalarFieldWriter()
{
}

void ScalarFieldWriter::execute()
{

    ScalarFieldHandle handle;
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

