//static char *id="@(#) $Id$";

#include "MPReader.h"
#include <Uintah/Datatypes/Particles/MPParticleGridReader.h>

#include <fstream>
#include <iostream> 
using std::endl;
#include <iomanip>
using std::setw;
#include <sstream>
using std::ostringstream;

#include <ctype.h>
#include <unistd.h>

namespace Uintah {
namespace Modules {

using namespace SCICore::Containers;

PSECore::Dataflow::Module* make_MPReader( const clString& id ) { 
  return new MPReader( id );
}


//--------------------------------------------------------------- 
MPReader::MPReader(const clString& id) 
  : Module("MPReader", id, Filter),
    filebase("filebase", id, this), animate("animate", id, this),
    startFrame("startFrame", id, this), endFrame("endFrame", id, this),
    increment("increment", id, this),
    tcl_status("tcl_status",id,this) 
  

{ 
      // Initialization code goes here 
  out=new ParticleGridReaderOPort(this,
				  "ParticleGridReader",
				  ParticleGridReaderIPort::Atomic);
  add_oport(out);
  animate.set(0);
  startFrame.set(0);
  endFrame.set(0);
  increment.set(0);
  if( filebase.get() != "" )
    need_execute = 1;
    

} 


//------------------------------------------------------------ 
MPReader::~MPReader(){} 

//-------------------------------------------------------------- 

bool MPReader::checkFile(const clString& fname)
{
    std::ifstream in( fname() );

  if ( !in ) {
    // TCL::execute( id + " errorDialog 'File doesn't exist'");
    cerr <<  "File doesn't exist"<<endl;
    in.close();
    return false;
  }
    
  clString head;
  
  in >> head;

  if( head != "MPD") {
    //TCL::execute( id + " errorDialog 'File has wrong format.'");
    cerr << "File has wrong format."<<endl;
    in.close();
    return false;
  } else {
    in.close();
    return true;
  }

}

void MPReader::execute() 
{ 

  tcl_status.set("Executing"); 
   // might have multiple filenames later for animations
   clString command( id + " activate");
   TCL::execute(command);
   //   command = id + " deselect";
   //TCL::execute(command);
   //cout <<endl;

   std::cerr<<"Filename = "<<filebase.get()<<endl;
   if( filebase.get() == "" )
     return;
   
   if( !animate.get() && checkFile( filebase.get() ) ) {
     tcl_status.set("Reading file");    
     reader = new MPParticleGridReader( filebase.get(), startFrame.get(),
					   endFrame.get(), increment.get());
     out->send( ParticleGridReaderHandle( reader ) );
   } else if ( animate.get() && checkFile( filebase.get() ) ) {
     tcl_status.set("Animating");    
     doAnimation();
   }
     tcl_status.set("Done");    

}

void MPReader::doAnimation()
{
  clString file = basename( filebase.get() );
  clString path = pathname( filebase.get() );
  const char *p = file();
  char n[5];
  char root[ 80 ];
  int i;
  int j = 0;
  int k = 0;
  for( i= 0; i < file.len(); i++ )
    {
      if(isdigit(*p)) n[j++] = *p;
      else root[k++] = *p;
      p++;
    }
  root[k] = '\0';

  for(i = startFrame.get(); i <= endFrame.get(); i += increment.get() ){
    ostringstream ostr;
    sleep(2);
    ostr.fill('0');
    ostr << path << "/"<< root<< setw(4)<<i;
    std::cerr << ostr.str()<< endl;
    reader = new MPParticleGridReader( ostr.str().c_str(), startFrame.get(),
					  endFrame.get(), increment.get() );
    filebase.set( ostr.str().c_str() );
    file = basename( filebase.get() );
    reset_vars();
    if( i != endFrame.get() && animate.get()){
      out->send_intermediate( ParticleGridReaderHandle( reader));
      tcl_status.set( file );
    }
    else {
      out->send(ParticleGridReaderHandle( reader));
      break;
    }
  }
  TCL::execute( id + " deselect");
}
  
//--------------------------------------------------------------- 
  

} // End namespace Modules
} // End namespace Uintah

//
// $Log$
// Revision 1.2  1999/10/07 02:08:31  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/21 16:12:26  kuzimmer
// changes made to support binary/ASCII file IO
//
// Revision 1.1  1999/07/27 17:08:58  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:11:10  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/06/09 23:23:44  kuzimmer
// Modified the modules to work with the new Material/Particle classes.  When a module needs to determine the type of particleSet that is incoming, the new stl dynamic type testing is used.  Works good so far.
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//
