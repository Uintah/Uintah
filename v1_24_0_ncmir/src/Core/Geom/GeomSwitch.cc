/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

