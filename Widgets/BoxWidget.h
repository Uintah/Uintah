
/*
 *  BoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Box_Widget_h
#define SCI_project_Box_Widget_h 1

#include <Widgets/BaseWidget.h>


class BoxWidget : public BaseWidget {
public:
   BoxWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   BoxWidget( const BoxWidget& );
   ~BoxWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   Vector GetAxis1();
   Vector GetAxis2();
   Vector GetAxis3();

   // Variable indexs
   enum { PointIULVar, PointIURVar, PointIDRVar, PointIDLVar,
	  PointOULVar, PointOURVar, PointODRVar, PointODLVar,
	  DistVar, HypoVar, DiagVar };

private:
   Vector oldaxis1, oldaxis2, oldaxis3;
};


#endif
