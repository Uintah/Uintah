
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


// Variable indexs
enum { GuageW_PointL, GuageW_PointR, GuageW_Dist, GuageW_Slider, GuageW_SDist, GuageW_Ratio};
// Material indexs
enum { GuageW_PointMatl, GuageW_EdgeMatl, GuageW_SliderMatl, GuageW_HighMatl };


class GuageWidget : public BaseWidget {
public:
   GuageWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   GuageWidget( const GuageWidget& );
   ~GuageWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline Real GetRatio();
   inline Vector GetAxis();

private:
   Vector oldaxis;
};


inline Real
GuageWidget::GetRatio()
{
   return (variables[GuageW_Ratio]->Get().x());
}


inline Vector
GuageWidget::GetAxis()
{
   Vector axis(variables[GuageW_PointR]->Get() - variables[GuageW_PointL]->Get());
   if (axis.length2() <= 1e-6)
      return oldaxis;
   else 
      return (oldaxis = axis.normal());
}


#endif
