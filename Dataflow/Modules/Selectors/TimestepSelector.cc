

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomGroup.h>

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

  DECLARE_MAKER(TimestepSelector)

//--------------------------------------------------------------- 
TimestepSelector::TimestepSelector(GuiContext* ctx) 
  : Module("TimestepSelector", ctx, Filter, "Selectors", "Uintah"),
    tcl_status(ctx->subVar("tcl_status")), 
    time(ctx->subVar("time")),
    max_time(ctx->subVar("max_time")),
    timeval(ctx->subVar("timeval")),
    animate(ctx->subVar("animate")),
    anisleep(ctx->subVar("anisleep")),
    archiveH(0),
    def_color_r_(ctx->subVar("def-color-r")),
    def_color_g_(ctx->subVar("def-color-g")),
    def_color_b_(ctx->subVar("def-color-b")),
    def_color_a_(ctx->subVar("def-color-a")),
    def_mat_handle_(scinew Material(Color(1.0, 1.0, 1.0)))
{ 
} 

//------------------------------------------------------------ 
TimestepSelector::~TimestepSelector(){} 
void TimestepSelector::execute() 
{ 
  tcl_status.set("Calling TimestepSelector!"); 
  in = (ArchiveIPort *) get_iport("Data Archive");
  out = (ArchiveOPort *) get_oport("Archive Timestep");
  ogeom=(GeometryOPort *) get_oport("Geometry");
  
  ArchiveHandle handle;
   if(!in->get(handle)){
     std::cerr<<"TimestepSelector::execute() Didn't get a handle\n";
     return;
   }

   if (archiveH.get_rep()  == 0 ){
     string visible;
     gui->eval(id + " isVisible", visible);
     if( visible == "0" ){
       gui->execute(id + " buildTopLevel");
     }
   }

   vector< double > times;
   vector< int > indices;
   try {
     DataArchive& archive = *((*(handle.get_rep()))());
     archive.queryTimesteps( indices, times );
     if( archiveH.get_rep() == 0 ||
	 archiveH.get_rep() != handle.get_rep()){
       gui->execute(id + " SetTimeRange " + to_string((int)times.size()));
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
     //DataArchive& archive = *((*(handle.get_rep()))());
     while( animate.get() && idx < (int)times.size() - 1){
       //       archive.purgeTimestepCache( times[idx] );
       DataArchive::cacheOnlyCurrentTimestep = true;
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

   // now ship the time out to the vis for those that are interested.
   GeomGroup *all = scinew GeomGroup();
   Point ref(14.0/16, 31.0/16, 0.0);
   Vector along(-0.5, -1.0, 0.0);
   char value[40];
   double v, hour, min, sec, microseconds;
   v =  times[idx];
   hour = trunc( v/3600.0);
   min = trunc( v/60.0 - (hour * 60));
   sec = trunc( v - (hour * 3600) - (min * 60));
   microseconds = (v - sec - (hour * 3600) - (min * 60))*(1e3);
   ostringstream oss;
   if(hour > 0 || min > 0 || sec > 0 ){
     oss.width(2);
     oss.fill('0');
     oss<<hour<<":";
     oss.width(2);
     oss.fill('0');
     oss<<min<<":";
     oss.width(2);
     oss.fill('0');
     oss<<sec<<" + ";
   }
   oss<<microseconds<<"ms";
   
   cerr<<"time is "<<v<<" hour = "<<hour<<
     ", min = "<<min<<", secs = "<<sec<<", microseconds = "<<
     microseconds<<", formatted: "<< oss.str()<<endl;
//    sprintf(value,"%.2d:%.2d:%.2g ", hour, min, sec);
//  sprintf(value,"%02d:%02d:%02.6g", hour, min, sec);
   all->add(scinew GeomText(oss.str(), ref + along,
				 def_mat_handle_->diffuse));
   GeomSticky *sticky = scinew GeomSticky(all);
   ogeom->delAll();
   ogeom->addObj(sticky, "TimeStamp");

   tcl_status.set("Done");
   // DumpAllocator(default_allocator, "timedump.allocator");
}

void
TimestepSelector::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2){
    args.error("TimestepSelector needs a minor command");
    return;
  }

  if (args[1] == "default_color_change") {
    def_color_r_.reset();
    def_color_g_.reset();
    def_color_b_.reset();
    Material *m = scinew Material(Color(def_color_r_.get(), def_color_g_.get(),
					def_color_b_.get()));
    def_mat_handle_ = m;
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace Uintah
