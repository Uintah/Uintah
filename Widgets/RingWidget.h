
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

enum { RingW_PointUL, RingW_PointUR, RingW_PointDR, RingW_PointDL,
       RingW_Dist, RingW_Hypo,
       RingW_Slider, RingW_SDist, RingW_Ratio };

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
   return (variables[RingW_Ratio]->Get().x());
}


inline const Vector&
RingWidget::GetAxis() const
{
   Vector axis(variables[RingW_PointDR]->Get() - variables[RingW_PointUL]->Get());
   if (axis.length() == 0.0)
      return Vector(0,0,0);
   else
      return axis.normal();
}


#endif
