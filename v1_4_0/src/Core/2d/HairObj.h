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
 *  HairObj.h: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_HairObj_h
#define SCI_HairObj_h 

#include <Core/2d/Widget.h>

namespace SCIRun {
  
class SCICORESHARE HairObj :  public Widget {
protected:
  double from_, to_;
  double pos_;

  double proj[16], model[16];
  int viewport[4];
public:
  
  HairObj( const string &name="hairline" );
  virtual ~HairObj();

  virtual void get_bounds( BBox2d & ) {} 
  double at() { return pos_; }
  virtual void recompute();
  virtual void select( double x, double y, int b );
  virtual void move( double x, double y, int b );
  virtual void release( double x, double y, int b );

  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false );
#endif
  static PersistentTypeID type_id;
  
  virtual void io(Piostream&);    
  
};

void Pio(Piostream&, HairObj*&);

} // namespace SCIRun

#endif // SCI_HairObj_h
