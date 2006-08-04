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
 *  PointLight.cc:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/PointLight.h>
#include <Core/Geom/GeomSphere.h>

namespace SCIRun {

Persistent* make_PointLight()
{
    return new PointLight("", Point(0,0,0), Color(0,0,0));
}

PersistentTypeID PointLight::type_id("PointLight", "Light", make_PointLight);

PointLight::PointLight(const string& name,
		       const Point& p, const Color& c, 
		       bool on, bool transformed )
: Light(name, on, transformed), p(p), c(c)
{
}

PointLight::~PointLight()
{
}

#define POINTLIGHT_VERSION 1

void PointLight::io(Piostream& stream)
{

    stream.begin_class("PointLight", POINTLIGHT_VERSION);
    // Do the base class first...
    Light::io(stream);
    Pio(stream, p);
    Pio(stream, c);
    stream.end_class();
}

} // End namespace SCIRun

