

#include <Packages/Uintah/Dataflow/Modules/Selectors/TimestepSelector.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomGroup.h>
#include <Dataflow/Network/Scheduler.h>


#include <sgi_stl_warnings_off.h>
#include <iostream> 
#include <sstream>
#include <string>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

using namespace SCIRun;
using namespace Uintah;

DECLARE_MAKER(TimestepSelector)

  //--------------------------------------------------------------- 

TimestepSelector::TimestepSelector(GuiContext* ctx) :
  Module("TimestepSelector", ctx, Filter, "Selectors", "Uintah"),
  tcl_status(ctx->subVar("tcl_status")), 
  time(ctx->subVar("time")),
  max_time(ctx->subVar("max_time")),
  timeval(ctx->subVar("timeval")),
  animate(ctx->subVar("animate")),
  tinc(ctx->subVar("tinc")),
  archiveH(0),
  def_color_r_(ctx->subVar("def-color-r")),
  def_color_g_(ctx->subVar("def-color-g")),
  def_color_b_(ctx->subVar("def-color-b")),
  def_color_a_(ctx->subVar("def-color-a")),
  def_mat_handle_(scinew Material(Color(1.0, 1.0, 1.0))),
  font_size_(ctx->subVar("font_size"))
{ 
} 

TimestepSelector::~TimestepSelector() {
  // Remove the callback
  sched->add_callback(network_finished, this);
} 

//------------------------------------------------------------ 

void
TimestepSelector::execute() 
{ 
  tcl_status.set("Calling TimestepSelector!"); 
  in = (ArchiveIPort *) get_iport("Data Archive");
  out = (ArchiveOPort *) get_oport("Archive Timestep");
  ogeom=(GeometryOPort *) get_oport("Geometry");
  
  ArchiveHandle handle;

  if (!(in->get(handle) && handle.get_rep())) {
    warning("Input field is empty.");
    return;
  }

  vector< double > times;
  vector< int > indices;
  DataArchiveHandle archive = handle->getDataArchive();
  try {
    archive->queryTimesteps( indices, times );
    if( archiveH.get_rep() == 0 || archiveH != handle){
      gui->execute(id + " SetTimeRange " + to_string((int)times.size()));
      archiveH = handle;
    }
  }
  catch (const InternalError& e) {
    cerr << "TimestepSelector caught exception: " << e.message() << endl;
    tcl_status.set("Exception");
    error(e.message());
    return;
  }
  
  // Check to see if the text color has changed.
  Color current_text_color( def_color_r_.get(), def_color_g_.get(), def_color_b_.get() );
  if( def_mat_handle_->diffuse != current_text_color ) {
    Material *m = scinew Material(Color(def_color_r_.get(), def_color_g_.get(),
                                        def_color_b_.get()));
    def_mat_handle_ = m;
  }

  // set the index for the correct timestep.
  unsigned int idx = time.get();
  // Do a bounds check
  if(idx >= times.size()) {
    idx = times.size()-1;
    time.set( idx );
  }

  // now set up time for dumping to vis

  GeomGroup *all;
  Point ref(14.0/16, 31.0/16, 0.0);
  Vector along(-0.5, -1.0, 0.0);
  double v, hour, min, sec, microseconds;

  if( animate.get() ) {
    // Make sure the caching is off
    archive->turnOffXMLCaching();

    // Get tinc and make sure it is a good value.
    int tinc_val = tinc.get();
    if (tinc_val < 0) {
      error("Time Step Increment is less than 0.");
      animate.set(0);
    } else {
      if (tinc_val == 0) {
        warning("Time Step Increment is equal to 0.  The time step will not increment.");
      }
      // See if incrementing idx will go out of bounds.  If it will,
      // don't change it.
      if (idx + tinc_val < times.size() ) {
        idx+=tinc_val;
        time.set( idx );
      } else {
        // Turn off animation
        animate.set(0);
        archive->turnOnXMLCaching();
      }
    }
  } else {
    archive->turnOnXMLCaching();
  }

  // Update the time in the display
  timeval.set(times[idx]);

  // Set the timestep in the archive and send it down
  handle->SetTimestep( idx );
  out->send(handle);


  // Generate the timestep geometry for the viewer
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

} // end execute()

// This is a callback made by the scheduler when the network finishes.
// It should ask for a reexecute if the module and increment the
// timestep if animate is on.
bool TimestepSelector::network_finished(void* ts_) {
  TimestepSelector* ts = (TimestepSelector*)ts_;
  ts->update_animate();
  return true;
}

void TimestepSelector::update_animate() {
  if( animate.get() ) {
    want_to_execute();
  }    
}

void TimestepSelector::set_context(Network* network) {
  Module::set_context(network);
  // Set up a callback to call after we finish
  sched->add_callback(network_finished, this);
}
