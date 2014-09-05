#include "ArchiveReader.h"
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/InternalError.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream> 
using std::endl;
using std::cerr;

namespace Uintah {

using namespace SCIRun;

  DECLARE_MAKER(ArchiveReader)

//--------------------------------------------------------------- 
  ArchiveReader::ArchiveReader(GuiContext* ctx)
  : Module("ArchiveReader", ctx, Filter, "DataIO", "Uintah"),
    filebase(ctx->subVar("filebase")), 
    tcl_status(ctx->subVar("tcl_status")), archiveH(0)
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
     archiveH = scinew Archive(reader);
   }

   out->send( archiveH );

}

  
//--------------------------------------------------------------- 
  
} // End namespace Uintah


