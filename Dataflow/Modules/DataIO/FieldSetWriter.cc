/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  FieldSetWriter.cc: Save persistent representation of a field to a file
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldSetPort.h>

namespace SCIRun {

class FieldSetWriter : public Module {
  GuiString filename_;
  GuiString filetype_;
public:
  FieldSetWriter(const clString& id);
  virtual ~FieldSetWriter();
  virtual void execute();
};

extern "C" Module* make_FieldSetWriter(const clString& id) {
  return new FieldSetWriter(id);
}

FieldSetWriter::FieldSetWriter(const clString& id)
  : Module("FieldSetWriter", id, Source, "DataIO", "SCIRun"),
    filename_("filename", id, this),
    filetype_("filetype", id, this)
{
}

FieldSetWriter::~FieldSetWriter()
{
}

void FieldSetWriter::execute()
{
  // Read data from the input port
  FieldSetIPort *inport = (FieldSetIPort *)get_iport(0);
  FieldSetHandle handle;
  if(!inport->get(handle))
    return;

  // If no name is provided, return
  clString fn(filename_.get());
  if(fn == "") {
    error("Warning: no filename in FieldSetWriter");
    return;
  }
   
  // Open up the output stream
  Piostream* stream;
  clString ft(filetype_.get());
  if(ft=="Binary"){
    stream=scinew BinaryPiostream(fn(), Piostream::Write);
  } else { // "ASCII"
    stream=scinew TextPiostream(fn(), Piostream::Write);
  }

  // Write the file
  Pio(*stream, handle);
  delete stream;
}

} // End namespace SCIRun
