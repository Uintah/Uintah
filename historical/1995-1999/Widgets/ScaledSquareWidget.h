
/*
 *  ScaledSquareWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_ScaledSquare_Widget_h
#define SCI_project_ScaledSquare_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { SSquareW_PointUL, SSquareW_PointUR, SSquareW_PointDR, SSquareW_PointDL,
       SSquareW_Dist, SSquareW_Hypo,
       SSquareW_Slider1, SSquareW_SDist1, SSquareW_Ratio1,
       SSquareW_Slider2, SSquareW_SDist2, SSquareW_Ratio2 };
// Material indexs
enum { SSquareW_PointMatl, SSquareW_EdgeMatl, SSquareW_SliderMatl, SSquareW_HighMatl };


class ScaledSquareWidget : public BaseWidget {
public:
   ScaledSquareWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ScaledSquareWidget( const ScaledSquareWidget& );
   ~ScaledSquareWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline Real GetRatio1() const;
   inline Real GetRatio2() const;
   inline Vector GetAxis1();
   inline Vector GetAxis2();

private:
   Vector oldaxis1, oldaxis2;
};


inline Real
ScaledSquareWidget::GetRatio1() const
{
   return (variables[SSquareW_Ratio1]->Get().x());
}


inline Real
ScaledSquareWidget::GetRatio2() const
{
   return (variables[SSquareW_Ratio2]->Get().x());
}


inline Vector
ScaledSquareWidget::GetAxis1()
{
   Vector axis(variables[SSquareW_PointUR]->Get() - variables[SSquareW_PointUL]->Get());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


inline Vector
ScaledSquareWidget::GetAxis2()
{
   Vector axis(variables[SSquareW_PointDL]->Get() - variables[SSquareW_PointUL]->Get());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


#endif
