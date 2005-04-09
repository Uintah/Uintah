
/*
 *  TensorFieldPort.cc
 *  
 *  Deals with making the TensorFieldPort - NOTE that since
 *  the TensorField is a templatized class and the SimpleIPort
 *  is also templatized AND has static members we have to
 *  explicity declare and set the static variables.  So for
 *  every type of TensorFieldPort that you want - you have
 *  to add the appropriate two lines down below.
 *
 *  Eric Lundberg, Oct 1998
 */

#include <Datatypes/TensorFieldPort.h>

clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
clString SimpleIPort<TensorFieldHandle>::port_color("green3");

