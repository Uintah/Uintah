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
 *  View.h:  The camera
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_View_h
#define SCI_Geom_View_h 1

#include <Util/Point.h>

namespace SemotusVisum {

typedef Vector3d Vector;

class View {
protected:
  Point3d eyep_;
  Point3d lookat_;
  Vector up_;
  double fov_;

public:
  View();
  View(const Point3d&, const Point3d&, const Vector&, double);
  View(const View&);
  ~View();
  View& operator=(const View&);
  
  // compare 2 views; are they exactly the same?
  int operator==(const View&);
  
  void get_viewplane(double aspect, double zdist,
		     Vector& u, Vector& v);
  
  void get_normalized_viewplane(Vector& u, Vector& v);
  
  Point3d eyespace_to_objspace(const Point3d& p, double aspect);
  Point3d objspace_to_eyespace(const Point3d& p, double aspect);
  
  Point3d eyespace_to_objspace_ns(const Point3d& p, double aspect);
  Point3d objspace_to_eyespace_ns(const Point3d& p, double aspect);
  
  double depth(const Point3d& p);
  
  Point3d eyep() const;
  void eyep(const Point3d&);
  Point3d lookat() const;
  void lookat(const Point3d&);
  Vector up() const;
  void up(const Vector&);
  double fov() const;	
  void fov(double);
  
};

}

#endif
