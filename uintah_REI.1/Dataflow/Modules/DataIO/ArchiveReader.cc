
#include <Packages/Uintah/Dataflow/Modules/DataIO/ArchiveReader.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/Exception.h>

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
    filebase(get_ctx()->subVar("filebase")), 
    tcl_status(get_ctx()->subVar("tcl_status")), archiveH(0),
    aName(""), aName_size(0)
{ 
  if( filebase.get() != "" )
    need_execute_ = true;
} 


//------------------------------------------------------------ 
ArchiveReader::~ArchiveReader(){} 

//-------------------------------------------------------------- 


void
ArchiveReader::execute() 
{ 
  struct stat statbuffer;

  if( filebase.get() == "" ) {
    warning( "No file specified. " );
    return;
  }

  tcl_status.set("Executing"); 
  out = (ArchiveOPort *) get_oport("Data Archive");

  string index( filebase.get() + "/index.xml" );
  errno = 0;
  int result = stat( index.c_str(), &statbuffer);
  
  if( result != 0 ) {
    char msg[1024];
    sprintf( msg, "ArchiveReader unable to open %s.  (Errno: %d)", index.c_str(), errno );
    error( msg );
    return;
  }

  if(filebase.get() != aName || aName_size != statbuffer.st_size) {
    try {
      reader = scinew DataArchive(filebase.get(),0,1,false);
    } catch ( const Exception& ex) {
      error("ArchiveReader caught exception: " + string(ex.message()));
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


