//static char *id="@(#) $Id$";

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

#include <Remote/Modules/remoteSalmon/remoteSalmon.h>

namespace Remote {
namespace Modules {

using PSECore::Dataflow::Module;

//----------------------------------------------------------------------
extern "C" Module* make_remoteSalmon(const clString& id) {
  return new remoteSalmon(id);
}


//----------------------------------------------------------------------
remoteSalmon::remoteSalmon(const clString& id)
: Salmon(id, "remoteSalmon")
{
}

//----------------------------------------------------------------------
void
remoteSalmon::tcl_command(TCLArgs& a, void* v) {
  Salmon::tcl_command(a, v);
}

} // End namespace Modules
} // End namespace Remote

