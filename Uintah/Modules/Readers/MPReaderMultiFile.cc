
#include "MPReaderMultiFile.h"
#include <Uintah/Datatypes/Particles/MFMPParticleGridReader.h>

#include <fstream>
#include <iostream> 
using std::endl;
#include <iomanip>
using std::setw;
#include <sstream>
#include <string>
using std::ostringstream;
using std::istringstream;
using std::string;
#include <vector>
using std::vector;
#include <algo.h>
using std::min;
using std::max;
#include <ctype.h>
#include <unistd.h>

namespace Uintah {
namespace Modules {

using namespace SCICore::Containers;

PSECore::Dataflow::Module* make_MPReaderMultiFile( const clString& id ) { 
  return new MPReaderMultiFile( id );
}


//--------------------------------------------------------------- 
MPReaderMultiFile::MPReaderMultiFile(const clString& id) 
  : Module("MPReaderMultiFile", id, Filter),
    filebase("filebase", id, this), dirbase("dirbase",id,this),
    timestep("timestep", id, this), animate("animate", id, this),
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
MPReaderMultiFile::~MPReaderMultiFile(){} 
//-------------------------------------------------------------- 
void MPReaderMultiFile::tcl_command(TCLArgs& args, void* userdata)
{

  if ( args[1] == "combineFiles") {
    if( args.count() !=3 ) {
      args.error("MPReaderMultiFile::combineFiles too few args.");
      return;
    }
    
    filebase.set( args[2]);
  
  } else {
    Module::tcl_command(args, userdata);
  }
}

//-------------------------------------------------------------- 
bool MPReaderMultiFile::checkFile(const clString& fname)
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

void MPReaderMultiFile::execute() 
{ 

  tcl_status.set("Executing"); 
   // might have multiple filenames later for animations
   clString command( id + " activate");
   TCL::execute(command);
   //   command = id + " deselect";
   //TCL::execute(command);
   //cout <<endl;

   //std::cerr<<"Filename = "<<filebase.get()<<endl;
   if( filebase.get() == "" )
     return;
   
   if( !animate.get() ) {
     tcl_status.set("Reading files"); 
     
     reader = new MFMPParticleGridReader( filebase.get(), startFrame.get(),
					   endFrame.get(), increment.get());
     out->send( ParticleGridReaderHandle( reader ) );
   } else if ( animate.get() ) {
     tcl_status.set("Animating");    
     //doAnimation();
   }
     tcl_status.set("Done");    

}

void MPReaderMultiFile::doAnimation()
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
    reader = new MFMPParticleGridReader( ostr.str().c_str(), startFrame.get(),
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

