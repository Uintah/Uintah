
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
#include <MUI.h>
#include <NotFinished.h>
#include <iostream.h>
#include <fstream.h>

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

Module* make_FieldReader()
{
    return new FieldReader;
}

Module* FieldReader::clone(int deep)
{
    return new FieldReader(*this, deep);
}

void FieldReader::execute()
{
    Piostream* stream=auto_istream(filename);
    if(!stream)
	return; // Can't open file...
    Field3DHandle field(new Field3D);
    // Read the file...
    field->io(*stream);
    delete stream;
    outfield->send_field(field);
}

void FieldReader::mui_callback(void*, int)
{
    want_to_execute();
}
