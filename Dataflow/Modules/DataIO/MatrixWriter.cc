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
 *  MatrixWriter.cc: Save persistent representation of a matrix to a file
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
#include <Dataflow/Ports/MatrixPort.h>

namespace SCIRun {

class MatrixWriter : public Module {
  MatrixIPort* inport_;
  GuiString filename_;
  GuiString filetype_;
  GuiInt split_;
public:
  MatrixWriter(const clString& id);
  virtual ~MatrixWriter();
  virtual void execute();
};

extern "C" Module* make_MatrixWriter(const clString& id) {
  return new MatrixWriter(id);
}

MatrixWriter::MatrixWriter(const clString& id)
  : Module("MatrixWriter", id, Source), filename_("filename", id, this),
    filetype_("filetype", id, this), split_("split", id, this)
{
  // Create the output port
  inport_=scinew MatrixIPort(this, "Persistent Data", MatrixIPort::Atomic);
  add_iport(inport_);
}

MatrixWriter::~MatrixWriter()
{
}

void MatrixWriter::execute()
{
  // Read data from the input port
  MatrixHandle handle;
  if(!inport_->get(handle))
    return;

  // If no name is provided, return
  clString fn(filename_.get());
  if(fn == "") {
    error("Warning: no filename in MatrixWriter");
    return;
  }

  // Open up the output stream
  Piostream* stream;
  clString ft(filetype_.get());
  if(ft=="Binary"){
    stream=scinew BinaryPiostream(fn, Piostream::Write);
  } else { // "ASCII"
    stream=scinew TextPiostream(fn, Piostream::Write);
  }

  // Check whether the file should be split into header and data
  handle->set_raw(split_.get());
  
  // Write the file
  Pio(*stream, handle);
  delete stream;
}

} // End namespace SCIRun
