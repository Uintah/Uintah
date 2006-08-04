

#include <Packages/Uintah/Dataflow/Modules/Selectors/TimestepSelector.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomLine.h>
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
  tcl_status(get_ctx()->subVar("tcl_status")), 
  time(get_ctx()->subVar("time")),
  max_time(get_ctx()->subVar("max_time")),
  timeval(get_ctx()->subVar("timeval")),
  animate(get_ctx()->subVar("animate")),
  tinc(get_ctx()->subVar("tinc")),
  in(0), out(0), ogeom(0), time_port(0),
  archiveH(0),
  def_color_r_(get_ctx()->subVar("def-color-r")),
  def_color_g_(get_ctx()->subVar("def-color-g")),
  def_color_b_(get_ctx()->subVar("def-color-b")),
  def_color_a_(get_ctx()->subVar("def-color-a")),
  def_mat_handle_(scinew Material(Color(1.0, 1.0, 1.0))),
  font_size_(get_ctx()->subVar("font_size")),
  timeposition_x(get_ctx()->subVar("timeposition_x")),
  timeposition_y(get_ctx()->subVar("timeposition_y")),
  timestep_text(0),
  timestep_geom_id(0),
  clock_geom_id(0),
  // Clock variables
  short_hand_res(get_ctx()->subVar("short_hand_res")),
  long_hand_res(get_ctx()->subVar("long_hand_res")),
  short_hand_ticks(get_ctx()->subVar("short_hand_ticks")),
  long_hand_ticks(get_ctx()->subVar("long_hand_ticks")),
  clock_position_x(get_ctx()->subVar("clock_position_x")),
  clock_position_y(get_ctx()->subVar("clock_position_y")),
  clock_radius(get_ctx()->subVar("clock_radius"))
{ 
} 

TimestepSelector::~TimestepSelector() {
  // Remove the callback
  sched_->remove_callback(network_finished, this);
} 

//------------------------------------------------------------ 

void
TimestepSelector::execute() 
{ 
  in = (ArchiveIPort *) get_iport("Data Archive");
  
  ArchiveHandle handle;

  if (!(in->get(handle) && handle.get_rep())) {
    warning("Input field is empty.");
    animate.set( 0 );
    return;
  }

  tcl_status.set("Calling TimestepSelector!"); 

  out = (ArchiveOPort *) get_oport("Archive Timestep");
  ogeom=(GeometryOPort *) get_oport("Geometry");
  time_port = (MatrixOPort *) get_oport("Timestep");

  vector< double > times;
  vector< int > indices;
  DataArchiveHandle archive = handle->getDataArchive();
  try {
    archive->queryTimesteps( indices, times );
    if( archiveH.get_rep() == 0 || archiveH != handle){
      get_gui()->execute(get_id() + " SetTimeRange " + to_string((int)times.size()));
      archiveH = handle;
    }
  }
  catch (const InternalError& e) {
    cerr << "TimestepSelector caught exception: " << e.message() << endl;
    tcl_status.set("Exception");
    error(e.message());
    return;
  }

  // Make sure we are dealing with a fresh set of GUI variables.
  reset_vars();
  
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

  // Stuff the goodies down the timestep port
  MatrixHandle time_matrix = scinew DenseMatrix(times.size()+2,1);
  // Stuff the current time into the first location
  time_matrix->put(0,0, times[idx]);
  // Stuff the current timestep index
  time_matrix->put(1,0, idx);
  // Stuff the rest of the timesteps in
  for (size_t i = 0; i < times.size(); ++i)
    time_matrix->put(i+2,0, times[i]);

  // Now send it down
  time_port->send(time_matrix);

  // Generate the timestep geometry for the viewer
  Point text_pos(timeposition_x.get(), timeposition_y.get(), 0.0);
  double hour, min, sec, microseconds;

  current_time =  times[idx];
#ifdef _WIN32
#define trunc (int)
#endif
  hour = trunc( current_time/3600.0);
  min = trunc( current_time/60.0 - (hour * 60));
  sec = trunc( current_time - (hour * 3600) - (min * 60));
  microseconds = (current_time - sec - (hour * 3600) - (min * 60))*(1e3);
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
  timestep_text = scinew GeomText(oss.str(), text_pos,
                                  def_mat_handle_->diffuse, 
                                  font_size_.get());
  GeomHandle sticky = scinew GeomSticky(timestep_text);
  // These geom ids start at 1, so 0 is an unintialized value
  ogeom->delAll();
  timestep_geom_id = ogeom->addObj(sticky, "TimeStamp Sticky");
  clock_geom_id = ogeom->addObj(createClock(current_time), "Clock Sticky");

  tcl_status.set("Done");
  // DumpAllocator(default_allocator, "timedump.allocator");

} // end execute()

void
TimestepSelector::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("GridVisualizer needs a minor command");
    return;
  }
  if(args[1] == "update_timeposition") {
    update_timeposition();
  }
  else if(args[1] == "update_clock") {
    update_clock();
  }
  else {
    Module::tcl_command(args, userdata);
  }
}

void
TimestepSelector::update_timeposition() {
  if (timestep_text.get_rep()) {
    // Make sure we get current GUI variable values.
    reset_vars();
    Point text_pos(timeposition_x.get(), timeposition_y.get(), 0.0);
    // Since we are storing timestep_text as a GeomObj handle we need
    // to reinterpret it back to a GeomText.
    GeomText* tt = dynamic_cast<GeomText*>(timestep_text.get_rep());
    ASSERT(tt);
    tt->moveTo(text_pos);
    // For some reason we have to delete the geometry and create a new
    // one for the changes to take effect.
    GeomHandle sticky = scinew GeomSticky(timestep_text);

    // Do we need to get the port?
    //    ogeom=(GeometryOPort *) get_oport("Geometry");

    // These geom ids start at 1, so 0 is an unintialized value
    if (timestep_geom_id != 0) {
      ogeom->delObj(timestep_geom_id);
    }
    timestep_geom_id = ogeom->addObj(sticky, "TimeStamp Sticky");
    ogeom->flush();
  }
}

void
TimestepSelector::update_clock() {
  if (clock_geom_id != 0 && ogeom != 0) {
    // These geom ids start at 1, so 0 is an unintialized value
    if (timestep_geom_id != 0) {
      ogeom->delObj(clock_geom_id);
    }
    clock_geom_id = ogeom->addObj(createClock(current_time), "Clock Sticky");
    ogeom->flush();
  }
}


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
  sched_->add_callback(network_finished, this);
}

// This will generate a clock given the number of seconds.
GeomHandle TimestepSelector::createClock(double num_seconds) {
  GeomGroup* all = scinew GeomGroup();
  reset_vars();
  double radius = clock_radius.get();
  Point center(clock_position_x.get(), clock_position_y.get(), 0);
  double offset = radius * M_SQRT1_2;

  // Draw rectangle around clock
  GeomLines* outline = scinew GeomLines();
  outline->add(center+Vector( offset,  offset, 0),
               center+Vector( offset, -offset, 0));
  outline->add(center+Vector( offset, -offset, 0),
               center+Vector(-offset, -offset, 0));
  outline->add(center+Vector(-offset, -offset, 0),
               center+Vector(-offset,  offset, 0));
  outline->add(center+Vector(-offset,  offset, 0),
               center+Vector( offset,  offset, 0));
  all->add(scinew GeomMaterial(outline,
                               scinew Material(Color(0,0,0),
                                               Color(1,1,1),
                                               Color(.5,.5,.5), 20)));

  // End rectangle
  GeomLines* sh_ticks = scinew GeomLines();
  int num_sh_ticks = short_hand_ticks.get();
  for(int i = 0; i < num_sh_ticks; ++i) {
    double theta = M_PI*2*(double)i/num_sh_ticks;
    Vector dir(sin(theta), cos(theta), 0);
    sh_ticks->add(center+dir*offset*0.8,
                  center+dir*offset*0.9);
  }
  all->add(scinew GeomMaterial(sh_ticks,
                               scinew Material(Color(0,0,0),
                                               Color(1,0,0),
                                               Color(.5,.5,.5), 20)));

  GeomLines* lh_ticks = scinew GeomLines();
  int num_lh_ticks = long_hand_ticks.get();
  for(int i = 0; i < num_lh_ticks; ++i) {
    double theta = M_PI*2*(double)i/num_lh_ticks;
    Vector dir(sin(theta), cos(theta), 0);
    lh_ticks->add(center+dir*offset*0.9,
                  center+dir*offset);
  }
  all->add(scinew GeomMaterial(lh_ticks,
                               scinew Material(Color(0,0,0),
                                               Color(1,1,1),
                                               Color(.5,.5,.5), 20)));

  double sh_theta = num_seconds/short_hand_res.get() * 2 * M_PI;
  GeomLine* short_hand =
    new GeomLine(center,
                 center+Vector(sin(sh_theta), cos(sh_theta), 0)*0.4*offset);
  short_hand->setLineWidth(3);
  all->add(scinew GeomMaterial(short_hand,
                               scinew Material(Color(0,0,0),
                                               Color(0.8,0,0),
                                               Color(.5,.5,.5), 20)));

  double lh_theta = num_seconds/long_hand_res.get() * 2 * M_PI;
  GeomLine* long_hand =
    new GeomLine(center,
                 center+Vector(sin(lh_theta), cos(lh_theta), 0)*0.7*offset);
  long_hand->setLineWidth(2);
  all->add(scinew GeomMaterial(long_hand,
                               scinew Material(Color(0,0,0),
                                               Color(0.8,0.8,0.8),
                                               Color(.5,.5,.5), 20)));


  return scinew GeomSticky(all);
}
