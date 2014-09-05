/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <Uintah/Dataflow/Modules/DataIO/ArchiveReader.h>
#include <Uintah/Core/DataArchive/DataArchive.h>
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


