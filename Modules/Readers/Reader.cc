
/*
 *  TYPEReader.cc: TYPE Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Readers/TYPEReader.h>
#include <ModuleList.h>
#include <NotFinished.h>

static Module* make_TYPEReader(const clString& id)
{
    return new TYPEReader(id);
}

#include "TYPERegister.h"

TYPEReader::TYPEReader(const clString& id)
: Module("TYPEReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=new TYPEOPort(this, "Output Data", TYPEIPort::Atomic);
    add_oport(outport);
}

TYPEReader::TYPEReader(const TYPEReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("TYPEReader::TYPEReader");
}

TYPEReader::~TYPEReader()
{
}

Module* TYPEReader::clone(int deep)
{
    return new TYPEReader(*this, deep);
}

void TYPEReader::execute()
{
    if(!handle.get_rep()){
	Piostream* stream=auto_istream(filename.get());
	if(!stream){
	    error(clString("Error reading file: ")+filename.get());
	    return; // Can't open file...
	}
	// Read the file...
	Pio(*stream, handle);
	if(!handle.get_rep()){
	    error("Error reading TYPE from file");
	    delete stream;
	    return;
	}
	delete stream;
    }
    outport->send(handle);
}
