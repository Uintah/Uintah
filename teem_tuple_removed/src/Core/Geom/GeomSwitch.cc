/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  Switch.cc:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */
#include <Core/Geom/GeomSwitch.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomSwitch()
{
    return scinew GeomSwitch(0,0);
}

PersistentTypeID GeomSwitch::type_id("GeomSwitch", "GeomObj", make_GeomSwitch);

Persistent* make_GeomTimeSwitch()
{
    return scinew GeomTimeSwitch(0,0,0);
}

PersistentTypeID GeomTimeSwitch::type_id("GeomTimeSwitch", "GeomObj", make_GeomTimeSwitch);

GeomSwitch::GeomSwitch(GeomHandle obj, int state)
: GeomContainer(obj), state(state)
{
}

GeomSwitch::GeomSwitch(const GeomSwitch& copy)
: GeomContainer(copy), state(copy.state)
{
}

GeomObj* GeomSwitch::clone()
{
    return scinew GeomSwitch(*this);
}

void GeomSwitch::set_state(int st)
{
   state=st;
}

int GeomSwitch::get_state()
{
   return state;
}

void GeomSwitch::get_bounds(BBox& bbox)
{
   if (state && child_.get_rep()) child_->get_bounds(bbox);
}


#define GEOMSWITCH_VERSION 1

void GeomSwitch::io(Piostream& stream)
{
    stream.begin_class("GeomSwitch", GEOMSWITCH_VERSION);
    GeomContainer::io(stream);
    Pio(stream, state);
    stream.end_class();
}



GeomTimeSwitch::GeomTimeSwitch(GeomHandle obj, double tbeg, double tend)
: GeomContainer(obj), tbeg(tbeg), tend(tend)
{
}

GeomTimeSwitch::GeomTimeSwitch(const GeomTimeSwitch& copy)
: GeomContainer(copy), tbeg(copy.tbeg), tend(copy.tend)
{
}

GeomObj* GeomTimeSwitch::clone()
{
    return scinew GeomTimeSwitch(*this);
}

#define GEOMTIMESWITCH_VERSION 1

void GeomTimeSwitch::io(Piostream& stream)
{
    stream.begin_class("GeomSwitch", GEOMTIMESWITCH_VERSION);
    GeomContainer::io(stream);
    Pio(stream, tbeg);
    Pio(stream, tend);
    stream.end_class();
}

} // End namespace SCIRun

