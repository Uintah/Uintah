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
 *  Light.cc: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/Light.h>

namespace SCIRun {

PersistentTypeID Light::type_id("Light", "Persistent", 0);

Light::Light(const string& name, bool on)
  : name(name), on(on)
{
}

Light::~Light()
{
}

#define LIGHT_VERSION 1

void Light::io(Piostream& stream)
{
    stream.begin_class("Light", LIGHT_VERSION);
    Pio(stream, name);
    stream.end_class();
}

void Pio(Piostream& stream, Light*& light)
{
    Persistent* tlight=light;
    stream.io(tlight, Light::type_id);
    if(stream.reading())
	light=(Light*)tlight;
}

} // End namespace SCIRun


