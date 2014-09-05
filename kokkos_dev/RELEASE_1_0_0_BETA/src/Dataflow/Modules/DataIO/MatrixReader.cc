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
 *  MatrixReader.cc: Read a persistent matrix from a file
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
#include <sys/stat.h>

namespace SCIRun {

class MatrixReader : public Module {
  MatrixOPort* outport_;
  GuiString filename_;
  MatrixHandle handle_;
  clString old_filename_;
  time_t old_filemodification_;
public:
  MatrixReader(const clString& id);
  virtual ~MatrixReader();
  virtual void execute();
};

extern "C" Module* make_MatrixReader(const clString& id) {
  return new MatrixReader(id);
}

MatrixReader::MatrixReader(const clString& id)
  : Module("MatrixReader", id, Source), filename_("filename", id, this),
    old_filemodification_(0)
{
  // Create the output port
  outport_=scinew MatrixOPort(this, "Output Data", MatrixIPort::Atomic);
  add_oport(outport_);
}

MatrixReader::~MatrixReader()
{
}

void MatrixReader::execute()
{
  clString fn(filename_.get());

  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if (stat(fn(), &buf)) {
    error(clString("Warning: couldn't get stats on file ")+fn);
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif
  if(!handle_.get_rep() || 
     fn != old_filename_ || 
     new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_=fn;
    Piostream* stream=auto_istream(fn);
    if(!stream){
      error(clString("Error reading file: ")+filename_.get());
      return;
    }
    
    // Read the file
    Pio(*stream, handle_);
    if(!handle_.get_rep() || stream->error()){
      error(clString("Error reading Matrix from file: ")+
	    filename_.get());
      delete stream;
      return;
    }
    delete stream;
  }
  
  // Send the data downstream
  outport_->send(handle_);
}

} // End namespace SCIRun
