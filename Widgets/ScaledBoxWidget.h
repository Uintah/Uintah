
/*
 *  ScaledBoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_ScaledBox_Widget_h
#define SCI_project_ScaledBox_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { SBoxW_PointIUL, SBoxW_PointIUR, SBoxW_PointIDR, SBoxW_PointIDL,
       SBoxW_PointOUL, SBoxW_PointOUR, SBoxW_PointODR, SBoxW_PointODL,
       SBoxW_Dist1, SBoxW_Dist2, SBoxW_Dist3, SBoxW_Hypo, SBoxW_Diag,
       SBoxW_Slider1, SBoxW_SDist1, SBoxW_Ratio1,
       SBoxW_Slider2, SBoxW_SDist2, SBoxW_Ratio2,
       SBoxW_Slider3, SBoxW_SDist3, SBoxW_Ratio3 };
// Material indexs
enum { SBoxW_PointMatl, SBoxW_EdgeMatl, SBoxW_SliderMatl, SBoxW_HighMatl };


class ScaledBoxWidget : public BaseWidget {
public:
   ScaledBoxWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ScaledBoxWidget( const ScaledBoxWidget& );
   ~ScaledBoxWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline Vector GetAxis1();
   inline Vector GetAxis2();
   inline Vector GetAxis3();

private:
   Vector oldaxis1, oldaxis2, oldaxis3;
};


inline Vector
ScaledBoxWidget::GetAxis1()
{
   Vector axis(variables[SBoxW_PointIUR]->Get() - variables[SBoxW_PointIUL]->Get());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


inline Vector
ScaledBoxWidget::GetAxis2()
{
   Vector axis(variables[SBoxW_PointIDL]->Get() - variables[SBoxW_PointIUL]->Get());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


inline Vector
ScaledBoxWidget::GetAxis3()
{
   Vector axis(variables[SBoxW_PointOUL]->Get() - variables[SBoxW_PointIUL]->Get());
   if (axis.length2() <= 1e-6)
      return oldaxis3;
   else
      return (oldaxis3 = axis.normal());
}


#endif
