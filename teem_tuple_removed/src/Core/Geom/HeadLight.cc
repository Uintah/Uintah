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
 *  HeadLight.cc:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/HeadLight.h>
#include <Core/Geom/View.h>

namespace SCIRun {

Persistent* make_HeadLight()
{
    return new HeadLight("", Color(0,0,0));
}

PersistentTypeID HeadLight::type_id("HeadLight", "Light", make_HeadLight);

HeadLight::HeadLight(const string& name, const Color& c, bool on)
: Light(name,on,false), c(c)
{
}

HeadLight::~HeadLight()
{
}

#define HEADLIGHT_VERSION 1

void HeadLight::io(Piostream& stream)
{

    stream.begin_class("HeadLight", HEADLIGHT_VERSION);
    // Do the base class first...
    Light::io(stream);
    Pio(stream, c);
    stream.end_class();
}


} // End namespace SCIRun

