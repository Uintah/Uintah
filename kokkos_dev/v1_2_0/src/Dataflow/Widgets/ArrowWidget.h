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
 *  ArrowWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Arrow_Widget_h
#define SCI_project_Arrow_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>

namespace SCIRun {
  class GeomPick;
}

namespace SCIRun {

  //using GeomPick;


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

class ArrowWidget : public BaseWidget {
public:
   ArrowWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ArrowWidget( const ArrowWidget& );
   virtual ~ArrowWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& );
   Point GetPosition() const;
   
   void SetLength( double );
   double GetLength();
   
   void SetDirection( const Vector& v );
   const Vector& GetDirection();

   virtual void widget_tcl( TCLArgs& );

   // Variable indexs         
   enum { PointVar, HeadVar, DistVar };

   // Material indexs
   enum { PointMatl, ShaftMatl, HeadMatl, ResizeMatl };

protected:
   virtual string GetMaterialName( const Index mindex ) const;   
   
private:
   Vector direction;
   double length;
};

} // End namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif


#endif
