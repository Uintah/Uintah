//static char *id="@(#) $Id$";

/*
 *  ParticleSetPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Uintah/Datatypes/Particles/ParticleSetPort.h>

//namespace Uintah {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace Uintah::Datatypes;

clString SimpleIPort<ParticleSetHandle>::port_type("ParticleSet");
clString SimpleIPort<ParticleSetHandle>::port_color("purple4");

//} // End namespace Datatypes
//} // End namespace Uintah

//
// $Log$
// Revision 1.2  1999/08/17 06:40:09  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:59:00  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//
