
/*
 *  BoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Box_Widget_h
#define SCI_project_Box_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { BoxW_PointIUL, BoxW_PointIUR, BoxW_PointIDR, BoxW_PointIDL,
       BoxW_PointOUL, BoxW_PointOUR, BoxW_PointODR, BoxW_PointODL,
       BoxW_Dist, BoxW_Hypo, BoxW_Diag };
// Material indexs
enum { BoxW_PointMatl, BoxW_EdgeMatl, BoxW_HighMatl };


class BoxWidget : public BaseWidget {
public:
   BoxWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   BoxWidget( const BoxWidget& );
   ~BoxWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline Vector GetAxis1();
   inline Vector GetAxis2();
   inline Vector GetAxis3();

private:
   Vector oldaxis1, oldaxis2, oldaxis3;
};


inline Vector
BoxWidget::GetAxis1()
{
   Vector axis(variables[BoxW_PointIUR]->GetPoint() - variables[BoxW_PointIUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


inline Vector
BoxWidget::GetAxis2()
{
   Vector axis(variables[BoxW_PointIDL]->GetPoint() - variables[BoxW_PointIUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


inline Vector
BoxWidget::GetAxis3()
{
   Vector axis(variables[BoxW_PointOUL]->GetPoint() - variables[BoxW_PointIUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis3;
   else
      return (oldaxis3 = axis.normal());
}


#endif
