

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include "TimestepSelector.h"
#include <iostream> 
#include <sstream>
#include <string>
#include <unistd.h>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace Uintah {
  
using namespace SCIRun;

extern "C" Module* make_TimestepSelector( const string& id ) {
  return scinew TimestepSelector( id ); 
}

//--------------------------------------------------------------- 
TimestepSelector::TimestepSelector(const string& id) 
  : Module("TimestepSelector", id, Filter),
    tcl_status("tcl_status", id, this), 
    animate("animate",id, this),
    anisleep("anisleep", id, this),
    time("time", id, this),
    timeval("timeval", id, this),
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
     std::cerr<<"TimestepSelector::execute() Didn't get a handle\n";
     return;
   }

   if (archiveH.get_rep()  == 0 ){
     string visible;
     TCL::eval(id + " isVisible", visible);
     if( visible == "0" ){
       TCL::execute(id + " buildTopLevel");
     }
   }

   vector< double > times;
   vector< int > indices;
   try {
     DataArchive& archive = *((*(handle.get_rep()))());
     archive.queryTimesteps( indices, times );
     if( archiveH.get_rep() == 0 ||
	 archiveH.get_rep() != handle.get_rep()){
       TCL::execute(id + " SetTimeRange " + to_string((int)times.size()));
       archiveH = handle;
     }
   }
   catch (const InternalError& e) {
     cerr << "TimestepSelector caught exception: " << e.message() << endl;
     tcl_status.set("Exception");
     return;
   }
  

   // what time is it?
   
   int t = time.get();

   // set the index for the correct timestep.
   int idx = 0;
   if(t < (int)times.size())
     idx = t;
   if(t >= (int)times.size())
     idx=(int)times.size()-1;

   timeval.set(times[idx]);

   if( animate.get() ){
     while( animate.get() && idx < (int)times.size() - 1){
       idx++;
       tcl_status.set( to_string( times[idx] ));
       time.set( idx );
       handle->SetTimestep( idx );
       out->send_intermediate( handle );
       sleep(unsigned( anisleep.get()));
       reset_vars();
     }
     animate.set(0);
   }
   handle->SetTimestep( idx );
   out->send(handle);
   tcl_status.set("Done");
}
} // End namespace Uintah
