
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

#include <PSECore/Widgets/BaseWidget.h>

namespace PSECore {
namespace Widgets {

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

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
} // End namespace PSECore

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif

//
// $Log$
// Revision 1.3  1999/10/07 02:07:25  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:38:32  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:08  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:25  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
