#include <Modules/CFD/TecplotFileSelector.h>
#include <Datatypes/TecplotReader.h>
#include <Classlib/NotFinished.h>
#include <Malloc/Allocator.h>
#include <iostream.h> 
#include <iomanip.h>
#include <strstream.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>


extern "C" { 
  
  Module* make_TecplotFileSelector(const clString& id) 
  { 
    return new TecplotFileSelector(id); 
  } 
  
}//extern 
  

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
    

} 

//---------------------------------------------------------- 
TecplotFileSelector::TecplotFileSelector(const TecplotFileSelector& copy,
					   int deep) 
  : Module(copy, deep), 
    filebase("filebase", id, this), animate("animate", id, this),
    startFrame("startFrame", id, this), endFrame("endFrame", id, this),
    increment("increment", id, this),
    tcl_status("tcl_status",id,this) 
{} 

//------------------------------------------------------------ 
TecplotFileSelector::~TecplotFileSelector(){} 

//------------------------------------------------------------- 
Module* TecplotFileSelector::clone(int deep) 
{ 
  return new TecplotFileSelector(*this, deep); 
} 
//-------------------------------------------------------------- 

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
   tcl_status.set("Calling TecplotFileSelector!"); 

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
     reader = scinew TecplotReader( filebase.get(), startFrame.get(),
				   endFrame.get(), increment.get());
   
     out->send( ParticleGridReaderHandle( reader ) );
   } else if ( animate.get() && checkFile( filebase.get() ) ) {
     doAnimation();
   }

}

void TecplotFileSelector::doAnimation()
{
  clString file = basename(  filebase.get() );
  clString path = pathname( filebase.get() );
  char *p = file();
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
    ostrstream ostr;
    
    ostr.fill('0');
    ostr << path << "/"<< root<< setw(4)<<i;
    cerr << ostr.str()<< endl;
    reader = scinew TecplotReader( ostr.str(), startFrame.get(),
				   endFrame.get(), increment.get() );
    filebase.set( ostr.str() );
    if( i != endFrame.get())
      out->send_intermediate( ParticleGridReaderHandle( reader));
    else
      out->send(ParticleGridReaderHandle( reader));
  }
  TCL::execute( id + " deselect");
}
  
//--------------------------------------------------------------- 
  
