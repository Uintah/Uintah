#include "ArchiveReader.h"
#include <Uintah/Interface/DataArchive.h>
#include <SCICore/Exceptions/InternalError.h>
#include <iostream> 
using std::endl;
using std::cerr;

namespace Uintah {
namespace Modules {

using SCICore::Containers::clString;
using PSECore::Datatypes::ArchiveIPort;
using PSECore::Datatypes::ArchiveOPort;

extern "C" PSECore::Dataflow::Module* make_ArchiveReader( const clString& id ) { 
  return scinew ArchiveReader( id );
}


//--------------------------------------------------------------- 
ArchiveReader::ArchiveReader(const clString& id) 
  : Module("ArchiveReader", id, Filter),
    filebase("filebase", id, this), 
    tcl_status("tcl_status",id,this) 
{ 
      // Initialization code goes here 
  out=scinew ArchiveOPort(this,
				  "ArchiveReader",
				  ArchiveIPort::Atomic);
  add_oport(out);
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

   std::cerr<<"Filename = "<<filebase.get()<<endl;
   if( filebase.get() == "" )
     return;

   if( string(filebase.get()()) != aName ){
     try {
       reader = scinew DataArchive( string(filebase.get()()) );
     } catch ( const SCICore::Exceptions::InternalError& ex) {
       cerr<<"Caught and exception\n";
       return;
     }
     aName = string(filebase.get()());
   }

   Archive *archive = scinew Archive( reader );
   out->send( archive );

}

  
//--------------------------------------------------------------- 
  

} // End namespace Modules
} // End namespace Uintah

