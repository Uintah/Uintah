
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

#include <PSECore/Widgets/BaseWidget.h>

namespace PSECore {
namespace Widgets {

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

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

} // End namespace Widgets
} // End namespace PSECore

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif

//
// $Log$
// Revision 1.3  1999/10/07 02:07:26  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:38:32  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:09  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:26  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//

#endif
