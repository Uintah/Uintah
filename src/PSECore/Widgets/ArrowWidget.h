
/*
 *  ArrowWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Arrow_Widget_h
#define SCI_project_Arrow_Widget_h 1

#include <Widgets/BaseWidget.h>

namespace SCICore {
  namespace GeomSpace {
    class GeomPick;
  }
}

namespace PSECommon {
namespace Widgets {

  //using SCICore::GeomSpace::GeomPick;

using PSECommon::Dataflow::Module;

class ArrowWidget : public BaseWidget {
public:
   ArrowWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ArrowWidget( const ArrowWidget& );
   virtual ~ArrowWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& );
   Point GetPosition() const;
   
   void SetDirection( const Vector& v );
   const Vector& GetDirection() const;

   virtual void widget_tcl( TCLArgs& );

   // Variable indexs
   enum { PointVar };

   // Material indexs
   enum { PointMatl, ShaftMatl, HeadMatl };

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Vector direction;
};

} // End namespace Widgets
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:56:05  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:22  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//


#endif
