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
 *  Hairline.h: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Hairline_h
#define SCI_Hairline_h 

#include <Core/GuiInterface/TclObj.h>
#include <Core/2d/DrawObj.h>
#include <Core/2d/HairObj.h>

namespace SCIRun {

class Diagram;  

class SCICORESHARE Hairline :  public TclObj, public HairObj {
private:
  Array1< DrawObj *> poly_;
  Diagram *parent_;

public:
  
  Hairline() : TclObj( "hairline"), HairObj("Hairline") {}
  Hairline( Diagram *, const string &name="Hairline" );
  virtual ~Hairline();

  void update();

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

void Pio(Piostream&, Hairline*&);

} // namespace SCIRun

#endif // SCI_Hairline_h
