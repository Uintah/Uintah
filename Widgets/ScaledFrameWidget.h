
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


// Variable indexs
enum { SFrameW_PointUL, SFrameW_PointUR, SFrameW_PointDR, SFrameW_PointDL,
       SFrameW_Dist1, SFrameW_Dist2, SFrameW_Hypo,
       SFrameW_Slider1, SFrameW_SDist1, SFrameW_Ratio1,
       SFrameW_Slider2, SFrameW_SDist2, SFrameW_Ratio2 };
// Material indexs
enum { SFrameW_PointMatl, SFrameW_EdgeMatl, SFrameW_SliderMatl, SFrameW_HighMatl };


class ScaledFrameWidget : public BaseWidget {
public:
   ScaledFrameWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ScaledFrameWidget( const ScaledFrameWidget& );
   ~ScaledFrameWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

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

private:
   Vector oldaxis1, oldaxis2;
};


#endif
