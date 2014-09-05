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
 *  CrosshairWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Crosshair_Widget_h
#define SCI_project_Crosshair_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>

namespace SCIRun {

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

class CrosshairWidget : public BaseWidget {
public:
   CrosshairWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   CrosshairWidget( const CrosshairWidget& );
   virtual ~CrosshairWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& );
   Point GetPosition() const;

   // They should be orthogonal.
   void SetAxes( const Vector& v1, const Vector& v2, const Vector& v3 );
   void GetAxes( Vector& v1, Vector& v2, Vector& v3 ) const;

   virtual void widget_tcl( TCLArgs& );

   // Variable indexs
   enum { CenterVar };

   // Material indexs
   enum { PointMatl, AxesMatl };
   
protected:
   virtual string GetMaterialName( const Index mindex ) const;   
   
private:
   Vector axis1, axis2, axis3;
};

} // End namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif


#endif
