
/*
 *  FieldReader.cc:  The first module!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <FieldReader/FieldReader.h>
#include <Field3D.h>
#include <Field3DPort.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_FieldReader()
{
    return new FieldReader;
}

static RegisterModule db1("Fields", "FieldReader", make_FieldReader);
static RegisterModule db2("Readers", "FieldReader", make_FieldReader);

FieldReader::FieldReader()
: UserModule("FieldReader", Source)
{
    // Create the output data handle and port
    outfield=new Field3DOPort(this, "Field", Field3DIPort::Atomic);
    add_oport(outfield);

    add_ui(new MUI_file_selection("IsoContour value", &filename,
				  MUI_widget::NotExecuting));
}

FieldReader::FieldReader(const FieldReader& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("FieldReader::FieldReader");
}

FieldReader::~FieldReader()
{
}

Module* FieldReader::clone(int deep)
{
    return new FieldReader(*this, deep);
}

void FieldReader::execute()
{
    if(!field_handle.get_rep()){
	Piostream* stream=auto_istream(filename);
	if(!stream){
	    error(clString("Error reading file: ")+filename);
	    return; // Can't open file...
	}
	field_handle=new Field3D;
	// Read the file...
	field_handle->io(*stream);
	delete stream;
    }
    outfield->send_field(field_handle);
}

void FieldReader::mui_callback(void*, int)
{
    field_handle=0;
    want_to_execute();
}
