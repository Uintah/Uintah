/*
 *  ColorMapReader.cc: Read a persistent colormap from a file
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
#include <Dataflow/Ports/ColorMapPort.h>
#include <sys/stat.h>

namespace SCIRun {

class ColorMapReader : public Module {
  ColorMapOPort* outport_;
  GuiString filename_;
  ColorMapHandle handle_;
  clString old_filename_;
  long old_filemodification_;
public:
  ColorMapReader(const clString& id);
  virtual ~ColorMapReader();
  virtual void execute();
};

extern "C" Module* make_ColorMapReader(const clString& id) {
  return new ColorMapReader(id);
}

ColorMapReader::ColorMapReader(const clString& id)
  : Module("ColorMapReader", id, Source), filename_("filename", id, this),
    old_filemodification_(0)
{
  // Create the output port
  outport_=scinew ColorMapOPort(this, "Output Data", ColorMapIPort::Atomic);
  add_oport(outport_);
}

ColorMapReader::~ColorMapReader()
{
}

void ColorMapReader::execute()
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
  if(!handle_.get_rep() || 
     fn != old_filename_ || 
     buf.st_mtim.tv_nsec != old_filemodification_)
  {
    old_filemodification_=buf.st_mtim.tv_nsec;
    old_filename_=fn;
    Piostream* stream=auto_istream(fn);
    if(!stream){
      error(clString("Error reading file: ")+filename_.get());
      return;
    }
    
    // Read the file
    Pio(*stream, handle_);
    if(!handle_.get_rep() || stream->error()){
      error(clString("Error reading ColorMap from file: ")+
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
