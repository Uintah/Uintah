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
 *  AxesObj.h: Displayable 2D object
 *
 *  Written by:
 *   Chris
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_AxesObj_h
#define SCI_AxesObj_h 

#include <Core/2d/HairObj.h>

namespace SCIRun {
  
class SCICORESHARE AxesObj :  public HairObj {

protected:
  double xpos_, ypos_;
  int num_h_tics_,num_v_tics_;

public:
  
  AxesObj( const string &name="axes" );
  virtual ~AxesObj();

  virtual void get_bounds( BBox2d & ) {} 
  double x_at() { return xpos_; }
  double y_at() { return ypos_; }
  void recompute();
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

void Pio(Piostream&, AxesObj*&);

} // namespace SCIRun

#endif // SCI_HairObj_h
