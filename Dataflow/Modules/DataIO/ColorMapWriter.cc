
/*
 *  ColorMapWriter.cc: ColorMap Writer class
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
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Datatypes/ColorMap.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {


class ColorMapWriter : public Module {
    ColorMapIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    ColorMapWriter(const clString& id);
    virtual ~ColorMapWriter();
    virtual void execute();
};

extern "C" Module* make_ColorMapWriter(const clString& id) {
  return new ColorMapWriter(id);
}

ColorMapWriter::ColorMapWriter(const clString& id)
: Module("ColorMapWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew ColorMapIPort(this, "Input Data", ColorMapIPort::Atomic);
    add_iport(inport);
}

ColorMapWriter::~ColorMapWriter()
{
}

void ColorMapWriter::execute()
{

    ColorMapHandle handle;
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

