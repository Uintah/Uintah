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
 *  PointWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Point_Widget_h
#define SCI_project_Point_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>
#include <Core/Geom/Material.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

namespace SCIRun {

class PointWidget : public BaseWidget {
public:
  PointWidget( Module* module, CrowdMonitor* lock, double widget_scale );
  PointWidget( const PointWidget& );
  virtual ~PointWidget();

  virtual void redraw();
  virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  void SetPosition( const Point& );
  Point GetPosition() const;

  virtual void widget_tcl( TCLArgs& );

  // Variable indexs
  enum { PointVar };

  // Material indexs
  enum { PointMatl };
   
protected:
  virtual string GetMaterialName( const Index mindex ) const;   
};


} // End namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif

#endif
