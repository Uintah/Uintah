//static char *id="@(#) $Id$";

#include <fstream>
using std::ifstream;
#include <iostream> 
using std::cerr;
using std::cout;
using std::endl;
#include <iomanip>
using std::setw;
#include <sstream>
using std::ostringstream;
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#include <SCICore/Malloc/Allocator.h>

#include "TecplotFileSelector.h"
#include "TecplotReader.h"

namespace Uintah {
namespace Modules {

using namespace SCICore::Containers;

//--------------------------------------------------------------- 
TecplotFileSelector::TecplotFileSelector(const clString& id) 
  : Module("TecplotFileSelector", id, Filter),
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
TecplotFileSelector::~TecplotFileSelector(){} 

//------------------------------------------------------------- 

bool TecplotFileSelector::checkFile(const clString& fname)
{
  ifstream in( fname() );

  if ( !in ) {
    // TCL::execute( id + " errorDialog 'File doesn't exist'");
    cerr <<  "File doesn't exist"<<endl;
    in.close();
    return false;
  }
    
  clString head;
  
  in >> head;

  if( head != "TITLE") {
    //TCL::execute( id + " errorDialog 'File has wrong format.'");
    cerr << "File has wrong format."<<endl;
    in.close();
    return false;
  } else {
    in.close();
    return true;
  }

}

void TecplotFileSelector::execute() 
{ 

  tcl_status.set("Executing"); 
   // might have multiple filenames later for animations
   clString command( id + " activate");
   TCL::execute(command);
   emit_vars(cout);
   cout << "Done!"<<endl; 
   //   command = id + " deselect";
   //TCL::execute(command);
   //cout <<endl;

   cerr<<filebase.get()<<endl;
   if( filebase.get() == "" )
     return;
   
   if( !animate.get() && checkFile( filebase.get() ) ) {
     tcl_status.set("Reading file");    
     reader = new TecplotReader( filebase.get(), startFrame.get(),
				   endFrame.get(), increment.get());
     out->send( ParticleGridReaderHandle( reader ) );
   } else if ( animate.get() && checkFile( filebase.get() ) ) {
     tcl_status.set("Animating");    
     doAnimation();
   }
     tcl_status.set("Done");    

}

void TecplotFileSelector::doAnimation()
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
    cerr << ostr.str()<< endl;
    reader = new TecplotReader( ostr.str().c_str(), startFrame.get(),
				   endFrame.get(), increment.get() );
    filebase.set( ostr.str().c_str() );
    if( i != endFrame.get())
      out->send_intermediate( ParticleGridReaderHandle( reader));
    else
      out->send(ParticleGridReaderHandle( reader));
  }
  TCL::execute( id + " deselect");
}
  
//--------------------------------------------------------------- 
  
PSECore::Dataflow::Module* make_TecplotFileSelector( const clString& id ) { 
  return new TecplotFileSelector( id );
}

} // End namespace Modules
} // End namespace Uintah

//
// $Log$
// Revision 1.6  1999/10/07 02:08:28  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/21 16:12:25  kuzimmer
// changes made to support binary/ASCII file IO
//
// Revision 1.4  1999/08/18 21:45:26  sparker
// Array1 const correctness, and subsequent fixes
// Array1 bug fix courtesy Tom Thompson
//
// Revision 1.3  1999/08/18 20:20:23  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:40:12  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCI-Run/Uintah.
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
