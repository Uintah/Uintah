//static char *id="@(#) $Id$";

/*
 *  VizGrid.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Uintah/Datatypes/Particles/VizGrid.h>

namespace Uintah {
namespace Datatypes {

PersistentTypeID VizGrid::type_id("VizGrid",
				  "Datatype", 0);

VizGrid::~VizGrid()
{
}

VizGrid::VizGrid()
{
}


#define VIZGRID_VERSION 1

void VizGrid::io(Piostream& stream)
{
  stream.begin_class("VizGrid", VIZGRID_VERSION);

  stream.end_class();
}

} // End namespace Datatypes
} // End namespace Uintah

