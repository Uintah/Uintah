
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

#include <Widgets/BaseWidget.h>


class ViewWidget : public BaseWidget {
public:
   ViewWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ViewWidget( const ViewWidget& );
   ~ViewWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   Vector GetAxis1();
   Vector GetAxis2();
   Point GetUL();
   Point GetUR();
   Point GetDR();
   Point GetDL();

   // Variable indexs
   enum { PointULVar, PointURVar, PointDRVar, PointDLVar,
	  EyeVar, ForeVar, ForeEyeVar, BackVar, BackEyeVar,
	  Dist1Var, Dist2Var, HypoVar, RatioVar };
   // Material indexs
   enum { PointMatl, EdgeMatl, SpecialMatl, HighMatl };
private:
   Vector oldaxis1;
   Vector oldaxis2;
};


#endif
