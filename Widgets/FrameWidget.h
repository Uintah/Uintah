
/*
 *  FrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Frame_Widget_h
#define SCI_project_Frame_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { FrameW_PointUL, FrameW_PointUR, FrameW_PointDR, FrameW_PointDL,
       FrameW_Dist1, FrameW_Dist2, FrameW_Hypo };
// Material indexs
enum { FrameW_PointMatl, FrameW_EdgeMatl, FrameW_HighMatl };


class FrameWidget : public BaseWidget {
public:
   FrameWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   FrameWidget( const FrameWidget& );
   ~FrameWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   void SetPosition( const Point& UL, const Point& UR, const Point& DL );
   void GetPosition( Point& UL, Point& UR, Point& DL );

   void SetPosition( const Point& center, const Vector& normal,
		     const Real size1, const Real size2 );
   void GetPosition( Point& center, Vector& normal,
		     Real& size1, Real& size2 );

   void SetSize( const Real size1, const Real size2 );
   void GetSize( Real& size1, Real& size2 ) const;
   
   Vector GetAxis1();
   Vector GetAxis2();

private:
   Vector oldaxis1, oldaxis2;
};


#endif
