
/*
 *  GaugeWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Gauge_Widget_h
#define SCI_project_Gauge_Widget_h 1

#include <Widgets/BaseWidget.h>


class GaugeWidget : public BaseWidget {
public:
   GaugeWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   GaugeWidget( const GaugeWidget& );
   virtual ~GaugeWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetRatio( const Real ratio );
   Real GetRatio() const;

   void SetEndpoints( const Point& end1, const Point& end2 );
   void GetEndpoints( Point& end1, Point& end2 ) const;

   const Vector& GetAxis();

   // Variable indexs
   enum { PointLVar, PointRVar, DistVar, SDistVar, RatioVar};

   // Material indexs
   enum { PointMatl, ShaftMatl, ResizeMatl, SliderMatl };

private:
   Vector oldaxis;
};


#endif
