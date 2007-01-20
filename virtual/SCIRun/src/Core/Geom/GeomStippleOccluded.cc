/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  GeomStippleOccluded: 
 *    Shows geometry w/ a stipple patter where it would
 *    normally be occluded because of the depth test
 *    Useful for widgets
 *
 *  Written by:
 *   McKay Davis
 *   SCI Institue
 *   University of Utah
 *   September 2005
 *
 *  Copyright (C) 2005 SCI Group
 */
#include <Core/Geom/GeomStippleOccluded.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

namespace SCIRun {

Persistent* make_GeomStippleOccluded()
{
    return scinew GeomStippleOccluded(0,0);
}

PersistentTypeID GeomStippleOccluded::type_id("GeomStippleOccluded", "GeomSwitch", make_GeomStippleOccluded);

GeomStippleOccluded::GeomStippleOccluded(GeomHandle obj, int state)
: GeomSwitch(obj, state)
{
}

GeomStippleOccluded::GeomStippleOccluded(const GeomStippleOccluded& copy)
: GeomSwitch(copy.child_, copy.state)
{
}

GeomObj* GeomStippleOccluded::clone()
{
    return scinew GeomStippleOccluded(*this);
}


#define GEOMSTIPPLEOCCLUDED_VERSION 1

void GeomStippleOccluded::io(Piostream& stream)
{
    stream.begin_class("GeomStippleOccluded", GEOMSTIPPLEOCCLUDED_VERSION);
    GeomContainer::io(stream);
    Pio(stream, state);
    stream.end_class();
}


} // End namespace SCIRun

