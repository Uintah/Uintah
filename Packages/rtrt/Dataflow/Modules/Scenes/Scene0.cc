/*
 *  Scene0.cc:  Real Time Ray Tracer rendering engine
 *
 *  Rendering engine of the real time ray tracer.  This module takes a scene  
 *  file from the input port and renders it.
 *
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <vector>
#include <iostream>
#include <float.h>
#include <time.h>
#include <stdlib.h>

namespace rtrt {

using namespace SCIRun;
using namespace std;

class Scene0 : public Module {
public:
  Scene0(const string& id);
  virtual ~Scene0();
  virtual void execute();
  void tcl_command(TCLArgs& args, void* userdata);

private:
};

static string widget_name("Scene0 Widget");
 
extern "C" Module* make_Scene0(const string& id) {
  return scinew Scene0(id);
}

Scene0::Scene0(const string& id)
: Module("Scene0", id, Filter)
{
  //  inColorMap = scinew ColorMapIPort( this, "ColorMap",
  //				     ColorMapIPort::Atomic);
  //  add_iport( inColorMap);
}

Scene0::~Scene0()
{
}

void Scene0::execute()
{
  reset_vars();
}

// This is called when the tcl code explicity calls a function besides
// needexecute.
void Scene0::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  else {
    Module::tcl_command(args, userdata);
  }
}


} // End namespace rtrt

