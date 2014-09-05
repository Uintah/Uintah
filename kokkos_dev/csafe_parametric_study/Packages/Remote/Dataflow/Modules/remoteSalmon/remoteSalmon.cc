
/*
 *  Salmon.cc:  The Geometry Viewer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/Remote/Dataflow/Modules/remoteSalmon/remoteSalmon.h>

namespace Remote {
using namespace SCIRun;

//----------------------------------------------------------------------
extern "C" Module* make_remoteSalmon(const clString& get_id()) {
  return new remoteSalmon(get_id());
}


//----------------------------------------------------------------------
remoteSalmon::remoteSalmon(const clString& get_id())
: Salmon(get_id(), "remoteSalmon")
{
}

//----------------------------------------------------------------------
void
remoteSalmon::tcl_command(TCLArgs& a, void* v) {
  Salmon::tcl_command(a, v);
}
} // End namespace Remote


