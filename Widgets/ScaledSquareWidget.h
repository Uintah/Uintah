
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

enum { SSquareW_PointUL, SSquareW_PointUR, SSquareW_PointDR, SSquareW_PointDL,
       SSquareW_Dist, SSquareW_Hypo,
       SSquareW_Slider1, SSquareW_SDist1, SSquareW_Ratio1,
       SSquareW_Slider2, SSquareW_SDist2, SSquareW_Ratio2 };

class ScaledSquareWidget : public BaseWidget {
public:
   ScaledSquareWidget( Module* module, double widget_scale );
   ScaledSquareWidget( const ScaledSquareWidget& );
   ~ScaledSquareWidget();

   virtual void execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline Real GetRatio1() const;
   inline Real GetRatio2() const;
   inline const Vector& GetAxis1() const;
   inline const Vector& GetAxis2() const;
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


inline const Vector&
ScaledSquareWidget::GetAxis1() const
{
   Vector axis(variables[SSquareW_PointUR]->Get() - variables[SSquareW_PointUL]->Get());
   return axis.normal();
}


inline const Vector&
ScaledSquareWidget::GetAxis2() const
{
   Vector axis(variables[SSquareW_PointDL]->Get() - variables[SSquareW_PointUL]->Get());
   return axis.normal();
}


#endif
