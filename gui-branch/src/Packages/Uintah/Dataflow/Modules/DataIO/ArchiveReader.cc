#include "ArchiveReader.h"
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Core/Exceptions/InternalError.h>
#include <sys/types.h>
#include <sys/stat.h>
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
  static int aName_size = 0;
  struct stat statbuffer;

  tcl_status.set("Executing"); 
  out = (ArchiveOPort *) get_oport("Data Archive");

   if( filebase.get() == "" )
     return;

   string index( filebase.get() + "/index.xml" );
   stat( index.c_str(), &statbuffer);

   if(filebase.get() != aName || aName_size != statbuffer.st_size) {
     try {
       reader = scinew DataArchive(filebase.get());
     } catch ( const InternalError& ex) {
       cerr << "ArchiveReader caught exception: " << ex.message() << endl;
       return;
     }
     aName = filebase.get();
     aName_size = statbuffer.st_size;
   }

   Archive *archive = scinew Archive( reader );
   out->send( archive );

}

  
//--------------------------------------------------------------- 
  
} // End namespace Uintah


