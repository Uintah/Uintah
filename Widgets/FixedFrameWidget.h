
/*
 *  FFrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Fixed_Frame_Widget_h
#define SCI_project_Fixed_Frame_Widget_h 1

#include <Widgets/BaseWidget.h>


class FixedFrameWidget : public BaseWidget {
public:
   FixedFrameWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   FixedFrameWidget( const FixedFrameWidget& );
   ~FixedFrameWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   inline Vector GetAxis1();
   inline Vector GetAxis2();

   // Variable indexs
   enum { PointULVar, PointURVar, PointDRVar, PointDLVar,
	  Dist1Var, Dist2Var, HypoVar, RatioVar };
   // Material indexs
   enum { PointMatl, EdgeMatl, HighMatl };
private:
   Vector oldaxis1;
   Vector oldaxis2;
};


inline Vector
FixedFrameWidget::GetAxis1()
{
   Vector axis(variables[PointURVar]->point() - variables[PointULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


inline Vector
FixedFrameWidget::GetAxis2()
{
   Vector axis(variables[PointDLVar]->point() - variables[PointULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


#endif
