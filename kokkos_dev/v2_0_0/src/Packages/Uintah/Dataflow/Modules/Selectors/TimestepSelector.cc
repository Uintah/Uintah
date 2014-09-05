

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Datatypes/Color.h>
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
    def_mat_handle_(scinew Material(Color(1.0, 1.0, 1.0))),
    font_size_(ctx->subVar("font_size"))
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

   // now set up time for dumping to vis

   GeomGroup *all;
   Point ref(14.0/16, 31.0/16, 0.0);
   Vector along(-0.5, -1.0, 0.0);
   double v, hour, min, sec, microseconds;

   if( animate.get() ){
     //DataArchive& archive = *((*(handle.get_rep()))());
     while( animate.get() ) { // && idx < (int)times.size() - 1){
       //       archive.purgeTimestepCache( times[idx] );
       DataArchive::cacheOnlyCurrentTimestep = true;
       idx++;
       if (idx == times.size()-1 ) break;  // idx = 0; //for continuous cycle
       tcl_status.set( to_string( times[idx] ));
       time.set( idx );
       handle->SetTimestep( idx );
       out->send_intermediate( handle );
       sleep(unsigned( anisleep.get()));
       reset_vars();
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
       all = scinew GeomGroup();
       all->add(scinew GeomText(oss.str(), ref + along,
				def_mat_handle_->diffuse, 
				font_size_.get()));
       GeomSticky *sticky = scinew GeomSticky(all);
       ogeom->delAll();
       ogeom->addObj(sticky, "TimeStamp");
       ogeom->flushViews();
       
     }
     animate.set(0);
   }
   handle->SetTimestep( idx );
   out->send(handle);


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
   all = scinew GeomGroup();
   all->add(scinew GeomText(oss.str(), ref + along,
			    def_mat_handle_->diffuse, 
			    font_size_.get()));
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
