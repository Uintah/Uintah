
/*
 *  PointWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Point_Widget_h
#define SCI_project_Point_Widget_h 1

#include <Widgets/BaseWidget.h>
#include <Geom/Material.h>

namespace PSECommon {
namespace Widgets {

class PointWidget : public BaseWidget {
public:
   PointWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   PointWidget( const PointWidget& );
   virtual ~PointWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& );
   Point GetPosition() const;

   virtual void widget_tcl( TCLArgs& );

   // Variable indexs
   enum { PointVar };

   // Material indexs
   enum { PointMatl };
   
protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
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
