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
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/GuiInterface/GuiVar.h> 
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

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

protected:
  
private:

  GuiString tcl_status;

  GuiInt time;
  GuiInt max_time;
  GuiDouble timeval;
  GuiInt animate;
  GuiInt anisleep;

  ArchiveIPort *in;
  ArchiveOPort *out;
  GeometryOPort *ogeom;
  
  ArchiveHandle archiveH;
  void setVars(ArchiveHandle ar);
  virtual void tcl_command(GuiArgs& args, void* userdata);
 
private:
  //! default color and material
  GuiDouble                def_color_r_;
  GuiDouble                def_color_g_;
  GuiDouble                def_color_b_;
  GuiDouble                def_color_a_;
  MaterialHandle           def_mat_handle_;
  
  GuiString                font_size_;

}; //class 
} // End namespace Uintah

#endif
