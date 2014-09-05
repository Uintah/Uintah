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
 *  ViewWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_View_Widget_h
#define SCI_project_View_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>
#include <Core/Geom/View.h>

namespace SCIRun {


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

class ViewWidget : public BaseWidget {
public:
   ViewWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	       const Real AspectRatio=1.3333);
   ViewWidget( const ViewWidget& );
   virtual ~ViewWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   View GetView();
   Vector GetUpVector();
   Real GetFOV() const;

   void SetView( const View& view );

   Real GetAspectRatio() const;
   void SetAspectRatio( const Real aspect );
   
   const Vector& GetEyeAxis();
   const Vector& GetUpAxis();
   Point GetFrontUL();
   Point GetFrontUR();
   Point GetFrontDR();
   Point GetFrontDL();
   Point GetBackUL();
   Point GetBackUR();
   Point GetBackDR();
   Point GetBackDL();

   // Variable indexs
   enum { EyeVar, ForeVar, LookAtVar, UpVar, UpDistVar, EyeDistVar, FOVVar };

   // Material indexs
   enum { EyesMatl, ResizeMatl, ShaftMatl, FrustrumMatl };
   
protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Real ratio;
   Vector oldaxis1;
   Vector oldaxis2;
};

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif

} // End namespace SCIRun


#endif
