
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

#include <PSECore/Widgets/BaseWidget.h>

namespace PSECore {
namespace Widgets {

class ScaledFrameWidget : public BaseWidget {
public:
   ScaledFrameWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ScaledFrameWidget( const ScaledFrameWidget& );
   virtual ~ScaledFrameWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& center, const Point& R, const Point& D );
   void GetPosition( Point& center, Point& R, Point& D );

   void SetPosition( const Point& center, const Vector& normal,
		     const Real size1, const Real size2 );
   void GetPosition( Point& center, Vector& normal,
		     Real& size1, Real& size2 );

   void SetRatioR( const Real ratio );
   Real GetRatioR() const;
   void SetRatioD( const Real ratio );
   Real GetRatioD() const;

   void SetSize( const Real sizeR, const Real sizeD );
   void GetSize( Real& sizeR, Real& sizeD ) const;
   
   const Vector& GetRightAxis();
   const Vector& GetDownAxis();

   // Variable indexs
   enum { CenterVar, PointRVar, PointDVar, DistRVar, DistDVar, HypoVar,
	  SDistRVar, RatioRVar, SDistDVar, RatioDVar };

   // Material indexs
   enum { PointMatl, EdgeMatl, ResizeMatl, SliderMatl };

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Vector oldrightaxis, olddownaxis;
};

} // End namespace Widgets
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:33  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:09  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:26  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
