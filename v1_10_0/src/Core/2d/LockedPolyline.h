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
 *  LockedPolyline.h: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_LockedPolyline_h
#define SCI_LockedPolyline_h 

#include <Core/Datatypes/Color.h>
#include <Core/Containers/Array1.h>
#include <Core/2d/Polyline.h>

namespace SCIRun {
  
class SCICORESHARE LockedPolyline : public Polyline {

public:
  LockedPolyline( const string &name="") : Polyline(name) {} 
  LockedPolyline( int i );
  LockedPolyline( const vector<double> &, const string &name="" );
  virtual ~LockedPolyline();

  double at(double);
  void add(double);

  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false );
#endif
  static PersistentTypeID type_id;

  virtual void io(Piostream&);    

};

void Pio(Piostream&, LockedPolyline*&);

} // namespace SCIRun

#endif // SCI_LockedPolyline_h
