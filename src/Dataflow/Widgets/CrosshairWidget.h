
/*
 *  CrosshairWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Crosshair_Widget_h
#define SCI_project_Crosshair_Widget_h 1

#include <PSECore/Widgets/BaseWidget.h>

namespace PSECore {
namespace Widgets {

class CrosshairWidget : public BaseWidget {
public:
   CrosshairWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   CrosshairWidget( const CrosshairWidget& );
   virtual ~CrosshairWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& );
   Point GetPosition() const;

   // They should be orthogonal.
   void SetAxes( const Vector& v1, const Vector& v2, const Vector& v3 );
   void GetAxes( Vector& v1, Vector& v2, Vector& v3 ) const;

   virtual void widget_tcl( TCLArgs& );

   // Variable indexs
   enum { CenterVar };

   // Material indexs
   enum { PointMatl, AxesMatl };
   
protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Vector axis1, axis2, axis3;
};

} // End namespace Widgets
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:29  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:06  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:24  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
