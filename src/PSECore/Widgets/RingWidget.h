
/*
 *  RingWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Ring_Widget_h
#define SCI_project_Ring_Widget_h 1

#include <Widgets/BaseWidget.h>

namespace PSECommon {
namespace Widgets {

class RingWidget : public BaseWidget {
public:
   RingWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   RingWidget( const RingWidget& );
   virtual ~RingWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& center, const Vector& normal, const Real radius );
   void GetPosition( Point& center, Vector& normal, Real& radius ) const;
   
   void SetRatio( const Real ratio );
   Real GetRatio() const;

   void GetPlane( Vector& v1, Vector& v2);

   void SetRadius( const Real radius );
   Real GetRadius() const;
   
   const Vector& GetRightAxis();
   const Vector& GetDownAxis();

   // Variable indexs
   enum { CenterVar, PointRVar, PointDVar, DistVar, HypoVar, Sqrt2Var,
	  SliderVar, AngleVar };

   // Materials indexs
   enum { PointMatl, RingMatl, SliderMatl, ResizeMatl, HalfResizeMatl };
   
protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Vector oldrightaxis, olddownaxis;
};

} // End namespace Widgets
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:56:08  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:25  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
