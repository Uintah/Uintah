
/*
 *  RingWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Ring_Widget_h
#define SCI_project_Ring_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { RingW_PointUL, RingW_PointUR, RingW_PointDR, RingW_PointDL,
       RingW_Dist, RingW_Hypo, RingW_Center,
       RingW_Slider, RingW_SDist, RingW_Angle, RingW_Const };
// Material indexs
enum { RingW_PointMatl, RingW_EdgeMatl, RingW_SliderMatl, RingW_SpecialMatl, RingW_HighMatl };


class RingWidget : public BaseWidget {
public:
   RingWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   RingWidget( const RingWidget& );
   ~RingWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   void SetPosition( const Point& center, const Vector& normal, const Real radius );
   void GetPosition( Point& center, Vector& normal, Real& radius ) const;
      
   void SetRatio( const Real ratio );
   Real GetRatio() const;

   void SetRadius( const Real radius );
   Real GetRadius() const;
   
   const Vector& GetAxis1();
   const Vector& GetAxis2();

private:
   Vector oldaxis1, oldaxis2;
};


#endif
