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
 *  PointLight.h:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_PointLight_h
#define SCI_Geom_PointLight_h 1

#include <Core/Geom/Light.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class SCICORESHARE PointLight : public Light {
    Point p;
    Color c;
public:
    PointLight(const string& name, const Point&, const Color&,
	       bool on = true, bool transformed = true);
    virtual ~PointLight();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx);
#endif
  void move( const Point& newP) { p = newP; }
  void setColor( const Color& newC) { c = newC; }
};

} // End namespace SCIRun


#endif /* SCI_Geom_PointLight_h */

