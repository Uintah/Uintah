
/*
 *  PIDLObjectPort.cc
 *  $Id$
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Uintah/Datatypes/Particles/PIDLObjectPort.h>

using namespace SCICore::Containers;
using namespace Uintah::Datatypes;

template<> clString SimpleIPort<PIDLObjectHandle>::port_type("PIDLObject");
template<> clString SimpleIPort<PIDLObjectHandle>::port_color("pink3");

//
// $Log$
// Revision 1.1  1999/10/07 02:08:24  sparker
// use standard iostreams and complex type
//
//
