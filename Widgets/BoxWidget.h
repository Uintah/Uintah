
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


class BoxWidget : public BaseWidget {
public:
   BoxWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	      Index aligned=0 );
   BoxWidget( const BoxWidget& );
   virtual ~BoxWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   const Vector& GetRightAxis();
   const Vector& GetDownAxis();
   const Vector& GetInAxis();

   // 0=no, 1=yes
   Index IsAxisAligned() const;
   void AxisAligned( const Index yesno );

   // Variable indexs
   enum { CenterVar, PointRVar, PointDVar, PointIVar,
	  DistRVar, DistDVar, DistIVar, HypoRDVar, HypoDIVar, HypoIRVar };

   // Material indexs
   enum { PointMatl, EdgeMatl, ResizeMatl };

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Index aligned;
   
   Vector oldrightaxis, olddownaxis, oldinaxis;
};


#endif
