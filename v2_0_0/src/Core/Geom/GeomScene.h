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
 *  GeomScene.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef GeomScene_h
#define GeomScene_h 1

#include <Core/share/share.h>

#include <Core/Persistent/Persistent.h>

#include <Core/Datatypes/Color.h>
#include <Core/Geom/View.h>

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {


class Lighting;
class GeomObj;

struct SCICORESHARE GeomScene : public Persistent {
    GeomScene();
    GeomScene(const Color& bgcolor, const View& view, Lighting* lighting,
	     GeomObj* topobj);
    Color bgcolor;
    View view;
    Lighting* lighting;
    GeomObj* top;

    virtual void io(Piostream&);
};

} // End namespace SCIRun


#endif // ifndef GeomScene_h

