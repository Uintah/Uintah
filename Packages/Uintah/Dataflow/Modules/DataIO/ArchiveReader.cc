#include "ArchiveReader.h"
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream> 
using std::endl;
using std::cerr;

namespace Uintah {

using namespace SCIRun;

extern "C" Module* make_ArchiveReader( const string& id ) { 
  return scinew ArchiveReader( id );
}

//--------------------------------------------------------------- 
ArchiveReader::ArchiveReader(const string& id) 
  : Module("ArchiveReader", id, Filter, "DataIO", "Uintah"),
    filebase("filebase", id, this), 
    tcl_status("tcl_status",id,this) 
{ 
  if( filebase.get() != "" )
    need_execute = 1;
} 


//------------------------------------------------------------ 
ArchiveReader::~ArchiveReader(){} 

//-------------------------------------------------------------- 


void ArchiveReader::execute() 
{ 
  static string aName("");
  tcl_status.set("Executing"); 
  out = (ArchiveOPort *) get_oport("Data Archive");
   std::cerr<<"Filename = "<<filebase.get()<<endl;
   if( filebase.get() == "" )
     return;

   if(filebase.get() != aName ) {
     try {
       reader = scinew DataArchive(filebase.get());
     } catch ( const InternalError& ex) {
       cerr << "ArchiveReader caught exception: " << ex.message() << endl;
       return;
     }
     aName = filebase.get();
   }

   Archive *archive = scinew Archive( reader );
   out->send( archive );

}

  
//--------------------------------------------------------------- 
  
} // End namespace Uintah


