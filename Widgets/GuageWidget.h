
/*
 *  GuageWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Guage_Widget_h
#define SCI_project_Guage_Widget_h 1

#include <Widgets/BaseWidget.h>


class GuageWidget : public BaseWidget {
public:
   GuageWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   GuageWidget( const GuageWidget& );
   ~GuageWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetRatio( const Real ratio );
   Real GetRatio() const;

   void SetEndpoints( const Point& end1, const Point& end2 );
   void GetEndpoints( Point& end1, Point& end2 ) const;

   const Vector& GetAxis();

   // Variable indexs
   enum { PointLVar, PointRVar, DistVar, SliderVar, SDistVar, RatioVar};
   // Material indexs
   enum { PointMatl, EdgeMatl, SliderMatl, ResizeMatl, HighMatl };
private:
   Vector oldaxis;
};


#endif
