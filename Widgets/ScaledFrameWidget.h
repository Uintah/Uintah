
/*
 *  ScaledFrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_ScaledFrame_Widget_h
#define SCI_project_ScaledFrame_Widget_h 1

#include <Widgets/BaseWidget.h>


class ScaledFrameWidget : public BaseWidget {
public:
   ScaledFrameWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ScaledFrameWidget( const ScaledFrameWidget& );
   ~ScaledFrameWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& UL, const Point& UR, const Point& DL );
   void GetPosition( Point& UL, Point& UR, Point& DL );

   void SetPosition( const Point& center, const Vector& normal,
		     const Real size1, const Real size2 );
   void GetPosition( Point& center, Vector& normal,
		     Real& size1, Real& size2 );

   void SetRatio1( const Real ratio );
   Real GetRatio1() const;
   void SetRatio2( const Real ratio );
   Real GetRatio2() const;

   void SetSize( const Real size1, const Real size2 );
   void GetSize( Real& size1, Real& size2 ) const;
   
   Vector GetAxis1();
   Vector GetAxis2();

   // Variable indexs
   enum { PointULVar, PointURVar, PointDRVar, PointDLVar,
	  Dist1Var, Dist2Var, HypoVar,
	  Slider1Var, SDist1Var, Ratio1Var,
	  Slider2Var, SDist2Var, Ratio2Var };

private:
   Vector oldaxis1, oldaxis2;
};


#endif
