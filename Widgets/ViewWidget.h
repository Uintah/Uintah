
/*
 *  ViewWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_View_Widget_h
#define SCI_project_View_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { ViewW_PointUL, ViewW_PointUR, ViewW_PointDR, ViewW_PointDL,
       ViewW_Eye, ViewW_Fore, ViewW_ForeEye, ViewW_Back, ViewW_BackEye,
       ViewW_Dist1, ViewW_Dist2, ViewW_Hypo, ViewW_Ratio };
// Material indexs
enum { ViewW_PointMatl, ViewW_EdgeMatl, ViewW_SpecialMatl, ViewW_HighMatl };


class ViewWidget : public BaseWidget {
public:
   ViewWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ViewWidget( const ViewWidget& );
   ~ViewWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline Vector GetAxis1();
   inline Vector GetAxis2();
   inline Point GetUL();
   inline Point GetUR();
   inline Point GetDR();
   inline Point GetDL();

private:
   Vector oldaxis1;
   Vector oldaxis2;
};


inline Vector
ViewWidget::GetAxis1()
{
   Vector axis(variables[ViewW_PointUR]->GetPoint() - variables[ViewW_PointUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


inline Vector
ViewWidget::GetAxis2()
{
   Vector axis(variables[ViewW_PointDL]->GetPoint() - variables[ViewW_PointUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


inline Point
ViewWidget::GetUL()
{
   Vector v(variables[ViewW_PointUL]->GetPoint() - variables[ViewW_Eye]->GetPoint());
   if (v.length2() <= 1e-6)
      return variables[ViewW_PointUL]->GetPoint(); // ?!
   else
      return (variables[ViewW_Eye]->GetPoint()
	      + v * (variables[ViewW_Back]->GetReal() / variables[ViewW_Fore]->GetReal()));
}


inline Point
ViewWidget::GetUR()
{
   Vector v(variables[ViewW_PointUR]->GetPoint() - variables[ViewW_Eye]->GetPoint());
   if (v.length2() <= 1e-6)
      return variables[ViewW_PointUR]->GetPoint(); // ?!
   else
      return (variables[ViewW_Eye]->GetPoint()
	      + v * (variables[ViewW_Back]->GetReal() / variables[ViewW_Fore]->GetReal()));
}


inline Point
ViewWidget::GetDR()
{
   Vector v(variables[ViewW_PointDR]->GetPoint() - variables[ViewW_Eye]->GetPoint());
   if (v.length2() <= 1e-6)
      return variables[ViewW_PointDR]->GetPoint(); // ?!
   else
      return (variables[ViewW_Eye]->GetPoint()
	      + v * (variables[ViewW_Back]->GetReal() / variables[ViewW_Fore]->GetReal()));
}


inline Point
ViewWidget::GetDL()
{
   Vector v(variables[ViewW_PointDL]->GetPoint() - variables[ViewW_Eye]->GetPoint());
   if (v.length2() <= 1e-6)
      return variables[ViewW_PointDL]->GetPoint(); // ?!
   else
      return (variables[ViewW_Eye]->GetPoint()
	      + v * (variables[ViewW_Back]->GetReal() / variables[ViewW_Fore]->GetReal()));
}


#endif
