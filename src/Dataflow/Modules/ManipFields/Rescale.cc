
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <iostream>

namespace SCIRun {

// an example implementation of a Field Manipulation function

extern "C" void Rescale(Module *m)
{
  // get handles to the individual ports
  dynamic_port_range* d = m->get_iport("Input Field");

  // get the GUI data
  clString x,y,z;
  TCL::eval("$page"+m->id+"Rescale.l1.factor get",x);
  TCL::eval("$page"+m->id+"Rescale.l2.factor get",y);
  TCL::eval("$page"+m->id+"Rescale.l3.factor get",z);

  postMessage(clString("Rescale factor = (")+x+","+y+","+z+")",false);

  //
  // put code that rescales a field here
  //

  return;
}

} // namespace SCIRun
