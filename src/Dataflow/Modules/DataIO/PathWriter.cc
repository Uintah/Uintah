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
 *  PathWriter.cc: Save persistent representation of a path to a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/PathPort.h>

namespace SCIRun {

class PathWriter : public Module {
  PathIPort* inport_;
  GuiString filename_;
  GuiString filetype_;
public:
  PathWriter(const string& id);
  virtual ~PathWriter();
  virtual void execute();
};

extern "C" Module* make_PathWriter(const string& id) {
  return new PathWriter(id);
}

PathWriter::PathWriter(const string& id)
  : Module("PathWriter", id, Source, "DataIO", "SCIRun"),
    filename_("filename", id, this),
    filetype_("filetype", id, this)
{
  // Create the output port
  inport_=scinew PathIPort(this, "Persistent Data", PathIPort::Atomic);
  add_iport(inport_);
}

PathWriter::~PathWriter()
{
}

void PathWriter::execute()
{
  // Read data from the input port
  PathHandle handle;
  if(!inport_->get(handle))
    return;

  // If no name is provided, return
  string fn(filename_.get());
  if(fn == "") {
    error("Warning: no filename in PathWriter");
    return;
  }
   
  // Open up the output stream
  Piostream* stream;
  string ft(filetype_.get());
  if(ft=="Binary"){
    stream=scinew BinaryPiostream(fn, Piostream::Write);
  } else { // "ASCII"
    stream=scinew TextPiostream(fn, Piostream::Write);
  }

  // Write the file
  Pio(*stream, handle);
  delete stream;
}

} // End namespace SCIRun
