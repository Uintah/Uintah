/*
 *  Packages/KurtScalarFieldReader.cc: ScalarField Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include "Packages/KurtScalarFieldReader.h"
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/String.h>
#include <fstream>
#include <iostream> 
using std::cerr;
using std::endl;
#include <iomanip>
using std::setw;
#include <sstream>
using std::ostringstream;
#include <string>
#include <unistd.h>

namespace Kurt {
using namespace SCIRun;


extern "C" Module* make_Packages/KurtScalarFieldReader(const clString& id) {
  return new Packages/KurtScalarFieldReader(id);
}

Packages/KurtScalarFieldReader::Packages/KurtScalarFieldReader(const clString& id)
: Module("Packages/KurtScalarFieldReader", id, Source), 
    filebase("filebase", id, this), animate("animate", id, this),
    startFrame("startFrame", id, this), endFrame("endFrame", id, this),
    increment("increment", id, this),
    tcl_status("tcl_status",id,this) 
{
    // Create the output data handle and port
    outport=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
    add_oport(outport);
    animate.set(0);
    startFrame.set(0);
    endFrame.set(0);
    increment.set(0);
    if( filebase.get() != "" )
      need_execute = 1;

}

Packages/KurtScalarFieldReader::~Packages/KurtScalarFieldReader()
{
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    Packages/KurtScalarFieldReader* reader=(Packages/KurtScalarFieldReader*)cbdata;
    if(TCLTask::try_lock()){
	// Try the malloc lock once before we call update_progress
	// If we can't get it, then back off, since our caller might
	// have it locked
	if(!Task::test_malloc_lock()){
	    TCLTask::unlock();
	    return;
	}
	reader->update_progress(pd);
	TCLTask::unlock();
    }
}
#endif

void Packages/KurtScalarFieldReader::execute()
{

    tcl_status.set("Executing"); 
    // might have multiple filenames later for animations
    clString command( id + " activate");

    if( filebase.get() == "" )
      return;

    TCL::execute(command);
    
    if( animate.get() ){
      tcl_status.set("Animating");
      if ( !doAnimation() )
	return;
    } else {
      tcl_status.set("Reading file");
      if( ! read(filebase.get()) )
	return;
    }
    tcl_status.set("Done"); 
    outport->send(handle); 
}

bool
Packages/KurtScalarFieldReader::read( const clString& fn)
{
    if(!handle.get_rep() || fn != old_filebase){
	old_filebase=fn;
	Piostream* stream=auto_istream(fn);
	if(!stream){
	    error(clString("Error reading file: ")+filebase.get());
	    return false; // Can't open file...
	}
	// Read the file...
//	stream->watch_progress(watcher, (void*)this);
	Pio(*stream, handle);
	if(!handle.get_rep() || stream->error()){
	    error("Error reading ScalarField from file");
	    delete stream;
	    return false;
	}
	delete stream;
    }
    return true;
}

bool 
Packages/KurtScalarFieldReader::doAnimation()
{
  int i;
  int filesRead = 0;
  clString file = basename( filebase.get() );
  clString path = pathname( filebase.get() );
  std::string f( file() );
  std::string root(f.begin(), f.begin() + (f.size()-4));


  for(i = startFrame.get(); i <= endFrame.get(); i += increment.get() ){
    ostringstream ostr;
    sleep(2);
    ostr.fill('0');
    ostr << path << "/"<< root.c_str()<< setw(4)<<i;
    std::cerr << ostr.str()<< endl;
    if( !read( ostr.str().c_str() ))
      continue;
    filesRead++;
    filebase.set( ostr.str().c_str() );
    file = basename( filebase.get() );
    reset_vars();
    if( i != endFrame.get() && animate.get()){
      outport->send_intermediate( handle);
      tcl_status.set( file );
    }
    if( !animate.get())
      break;
    }
  TCL::execute( id + " deselect");
  if( filesRead )
    return 1;
  else
    return 0;
}

} // End namespace Kurt


