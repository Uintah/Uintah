
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
#include <MUI.h>
#include <ModuleList.h>
#include <NotFinished.h>

static Module* make_TYPEReader()
{
    return new TYPEReader;
}

#include "TYPERegister.h"

TYPEReader::TYPEReader()
: UserModule("TYPEReader", Source)
{
    // Create the output data handle and port
    outport=new TYPEOPort(this, "Output Data", TYPEIPort::Atomic);
    add_oport(outport);

    add_ui(new MUI_file_selection("TYPE file", &filename,
				  MUI_widget::NotExecuting));
}

TYPEReader::TYPEReader(const TYPEReader& copy, int deep)
: UserModule(copy, deep)
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
	Piostream* stream=auto_istream(filename);
	if(!stream){
	    error(clString("Error reading file: ")+filename);
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

void TYPEReader::mui_callback(void*, int)
{
    handle=0;
    want_to_execute();
}
