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
 *  Lighting.cc:  The light sources in a scene
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/Lighting.h>
#include <Core/Geom/Light.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {

Lighting::Lighting()
: amblight(Color(0,0,0))
{
}

Lighting::~Lighting()
{
}

#define LIGHTING_VERSION 1

void Pio( Piostream& stream, Lighting & l )
{

    stream.begin_class("Lighting", LIGHTING_VERSION);
    Pio(stream, l.lights);
    Pio(stream, l.amblight);
    stream.end_class();
}

} // End namespace SCIRun

