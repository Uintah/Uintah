
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


class ScaledBoxWidget : public BaseWidget {
public:
   ScaledBoxWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ScaledBoxWidget( const ScaledBoxWidget& );
   virtual ~ScaledBoxWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   Real GetRatio1() const;
   Real GetRatio2() const;
   Real GetRatio3() const;

   Vector GetAxis1();
   Vector GetAxis2();
   Vector GetAxis3();

   // Variable indexs
   enum { PointIULVar, PointIURVar, PointIDRVar, PointIDLVar,
	  PointOULVar, PointOURVar, PointODRVar, PointODLVar,
	  Dist1Var, Dist2Var, Dist3Var, HypoVar, DiagVar,
	  Slider1Var, SDist1Var, Ratio1Var,
	  Slider2Var, SDist2Var, Ratio2Var,
	  Slider3Var, SDist3Var, Ratio3Var };

private:
   Vector oldaxis1, oldaxis2, oldaxis3;
};


#endif
