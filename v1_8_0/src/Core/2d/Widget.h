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
  University of Utah. All Rightsget_iports(name Reserved.
*/


/*
 *  Widget.h: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Widget_h
#define SCI_Widget_h 

#include <Core/2d/DrawObj.h>

namespace SCIRun {
  
class SCICORESHARE Widget :  public DrawObj {
protected:
  BBox2d bbox_;

public:
  Widget( const string &name="" );
  virtual ~Widget();

  virtual void select( double , double , int  ) {}
  virtual void move( double , double , int  ) {}
  virtual void release( double , double , int ) {}

  virtual void get_bounds(BBox2d&) {}
  
  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false) {}
#endif
  static PersistentTypeID type_id;
  
  virtual void io(Piostream&);    
  
};

void Pio(Piostream&, Widget*&);

} // namespace SCIRun

#endif // SCI_Widget_h
