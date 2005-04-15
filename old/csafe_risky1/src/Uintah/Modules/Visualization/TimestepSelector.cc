

#include <SCICore/Util/NotFinished.h>
#include <Uintah/Interface/DataArchive.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include "TimestepSelector.h"
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace Uintah {
namespace Modules {

using SCICore::Containers::to_string;
using namespace SCICore::TclInterface;
using SCICore::Geometry::BBox;
using namespace Uintah::Datatypes;
using namespace PSECore::Datatypes;

extern "C" Module* make_TimestepSelector( const clString& id ) {
  return scinew TimestepSelector( id ); 
}

//--------------------------------------------------------------- 
TimestepSelector::TimestepSelector(const clString& id) 
  : Module("TimestepSelector", id, Filter),
    tcl_status("tcl_status", id, this), 
    time("time", id, this),
    timeval("timeval", id, this),
    animate("animate",id, this),
    archiveH(0)
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=new ArchiveIPort(this, "Data Archive",
		      ArchiveIPort::Atomic);
  out=new ArchiveOPort(this, "Archive Timestep",
			       ArchiveIPort::Atomic);
  // Add them to the Module
  add_iport(in);
  add_oport(out);

} 

//------------------------------------------------------------ 
TimestepSelector::~TimestepSelector(){} 
void TimestepSelector::execute() 
{ 
  tcl_status.set("Calling TimestepSelector!"); 
  
  ArchiveHandle handle;
   if(!in->get(handle)){
     std::cerr<<"Didn't get a handle\n";
     return;
   }

   if (archiveH.get_rep()  == 0 ){
     clString visible;
     TCL::eval(id + " isVisible", visible);
     if( visible == "0" ){
       TCL::execute(id + " buildTopLevel");
     }
   }

   DataArchive& archive = *((*(handle.get_rep()))());

   vector< double > times;
   vector< int > indices;
   archive.queryTimesteps( indices, times );
   if( archiveH.get_rep() == 0 ||
       archiveH.get_rep() != handle.get_rep()){
     TCL::execute(id + " SetTimeRange " + to_string((int)times.size()));
     archiveH = handle;
   }

   

   // what time is it?
   
   int t = time.get();

   // set the index for the correct timestep.
   int idx = 0;
   if(t < times.size())
     idx = t;
   if(t >= times.size())
     idx=times.size()-1;

   timeval.set(times[idx]);

   if( animate.get() ){
     while( animate.get() && idx < times.size() - 1){
       idx++;
       tcl_status.set( to_string( times[idx] ));
       time.set( idx );
       handle->SetTimestep( idx );
       out->send_intermediate( handle );
       reset_vars();
     }
     animate.set(0);
   }
   handle->SetTimestep( idx );
   out->send(handle);
   tcl_status.set("Done");
}

} // end namespace Modules
} // end namespace Uintah
  
