
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

enum { GuageW_PointL, GuageW_PointR, GuageW_Dist, GuageW_Slider, GuageW_SDist, GuageW_Ratio};

class GuageWidget : public BaseWidget {
public:
   GuageWidget( Module* module, double widget_scale );
   GuageWidget( const GuageWidget& );
   ~GuageWidget();

   virtual void execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline Real GetRatio() const;
   inline const Vector& GetAxis() const;
};


inline Real
GuageWidget::GetRatio() const
{
   return (variables[GuageW_Ratio]->Get().x());
}


inline const Vector&
GuageWidget::GetAxis() const
{
   Vector axis(variables[GuageW_PointR]->Get() - variables[GuageW_PointL]->Get());
   if (axis.length() == 0.0)
      return Vector(0,0,0);
   else
      return axis.normal();
}


#endif
