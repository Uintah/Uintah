/****************************************
CLASS
    TimestepSelector


OVERVIEW TEXT
    This module receives a DataArchive and selects the visualized timestep.
    Or Animates the data.



KEYWORDS
    

AUTHOR
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June 2000

    Copyright (C) 1999 SCI Group

LOG
    Created June 26, 2000
****************************************/
#ifndef TIMESTEPSELECTOR_H
#define TIMESTEPSELECTOR_H 1


#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Dataflow/Network/Module.h> 
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/GuiInterface/GuiVar.h> 
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

using namespace SCIRun;

class TimestepSelector : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  TimestepSelector(GuiContext* ctx);

  // GROUP: Destructors
  //////////
  virtual ~TimestepSelector(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute();

  // Accept TCL commands
  virtual void tcl_command(GuiArgs& args, void* userdata);

  // This is a callback made by the scheduler when the network
  // finishes.  It should ask for a reexecute if the module and
  // increment the timestep if animate is on.
  static bool network_finished(void* ts_);

  void update_animate();

  // Inherited from Module.  I need this to setup the callback
  // function for the scheduler.
  virtual void set_context(Network* network);
protected:
  
private:

  GuiString tcl_status;

  GuiInt time;
  GuiInt max_time;
  GuiDouble timeval;
  GuiInt animate;
  GuiInt tinc;

  ArchiveIPort *in;
  ArchiveOPort *out;
  GeometryOPort *ogeom;
  MatrixOPort  *time_port;
  
  ArchiveHandle archiveH;
  void setVars(ArchiveHandle ar);

private:
  //! default color and material
  GuiDouble         def_color_r_;
  GuiDouble         def_color_g_;
  GuiDouble         def_color_b_;
  GuiDouble         def_color_a_;
  MaterialHandle    def_mat_handle_;
  
  GuiString         font_size_;

  GuiDouble         timeposition_x;
  GuiDouble         timeposition_y;

  GeomHandle        timestep_text;
  int               timestep_geom_id;
  void update_timeposition();

  //////////////////
  //  Clock variables

  int               clock_geom_id;
  double            current_time;
  
  // Amount of time one revolution represents
  GuiDouble         short_hand_res;
  GuiDouble         long_hand_res;
  // Number of ticks to display going around
  GuiInt            short_hand_ticks;
  GuiInt            long_hand_ticks;
  // Position and size of the clock
  GuiDouble         clock_position_x;
  GuiDouble         clock_position_y;
  GuiDouble         clock_radius;

  // This will generate a clock given the number of seconds.
  GeomHandle createClock(double num_seconds);
  void update_clock();
}; //class 
} // End namespace Uintah

#endif
