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
 *  DirectionalLight.cc:  A Directional light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/DirectionalLight.h>
#include <Core/Geom/GeomSphere.h>

namespace SCIRun {

Persistent* make_DirectionalLight()
{
    return new DirectionalLight("", Vector(0,0,0), Color(0,0,0));
}

PersistentTypeID DirectionalLight::type_id("DirectionalLight", "Light", make_DirectionalLight);

DirectionalLight::DirectionalLight(const string& name,
				   const Vector& v, const Color& c, 
				   bool on, bool transformed)
: Light(name, on, transformed), v(v), c(c)
{
}

DirectionalLight::~DirectionalLight()
{
}

#define POINTLIGHT_VERSION 1

void DirectionalLight::io(Piostream& stream)
{

    stream.begin_class("DirectionalLight", POINTLIGHT_VERSION);
    // Do the base class first...
    Light::io(stream);
    Pio(stream, v);
    Pio(stream, c);
    stream.end_class();
}

} // End namespace SCIRun

