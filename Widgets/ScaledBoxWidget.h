
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
   ScaledBoxWidget( Module* module, CrowdMonitor* lock, double widget_scale,
		   Index aligned=0 );
   ScaledBoxWidget( const ScaledBoxWidget& );
   virtual ~ScaledBoxWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& center, const Point& R, const Point& D,
		     const Point& I );
   void GetPosition( Point& center, Point& R, Point& D, Point& I );

   void SetRatioR( const Real ratio );
   Real GetRatioR() const;
   void SetRatioD( const Real ratio );
   Real GetRatioD() const;
   void SetRatioI( const Real ratio );
   Real GetRatioI() const;

   const Vector& GetRightAxis();
   const Vector& GetDownAxis();
   const Vector& GetInAxis();

   // 0=no, 1=yes
   Index IsAxisAligned() const;
   void AxisAligned( const Index yesno );

   // Variable indexs
   enum { CenterVar, PointRVar, PointDVar, PointIVar,
	  DistRVar, DistDVar, DistIVar, HypoRDVar, HypoDIVar, HypoIRVar,
	  SDistRVar, RatioRVar, SDistDVar, RatioDVar, SDistIVar, RatioIVar };

   // Material indexs
   enum { PointMatl, EdgeMatl, ResizeMatl, SliderMatl };

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Index aligned;

   Vector oldrightaxis, olddownaxis, oldinaxis;
};


#endif
