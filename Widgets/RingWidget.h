
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
       RingW_Slider, RingW_SDist, RingW_Angle };
// Material indexs
enum { RingW_PointMatl, RingW_EdgeMatl, RingW_SliderMatl, RingW_SpecialMatl, RingW_HighMatl };


class RingWidget : public BaseWidget {
public:
   RingWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   RingWidget( const RingWidget& );
   ~RingWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline Real GetRatio() const;
   inline const Vector& GetAxis() const;
};


inline Real
RingWidget::GetRatio() const
{
   return (variables[RingW_Angle]->Get().x() + 3.14159) / (2.0 * 3.14159);
}


inline const Vector&
RingWidget::GetAxis() const
{
   static Vector oldaxis;
   Vector axis(variables[RingW_PointDR]->Get() - variables[RingW_PointUL]->Get());
   if (axis.length2() <= 1e-6)
      return oldaxis;
   else
      return (oldaxis = axis.normal());
}


#endif
