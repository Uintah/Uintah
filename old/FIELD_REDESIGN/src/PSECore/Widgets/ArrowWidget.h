
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

#include <PSECore/Widgets/BaseWidget.h>

namespace SCICore {
  namespace GeomSpace {
    class GeomPick;
  }
}

namespace PSECore {
namespace Widgets {

  //using SCICore::GeomSpace::GeomPick;

using PSECore::Dataflow::Module;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

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
   
   void SetLength( double );
   double GetLength();
   
   void SetDirection( const Vector& v );
   const Vector& GetDirection();

   virtual void widget_tcl( TCLArgs& );

   // Variable indexs         
   enum { PointVar, HeadVar, DistVar };

   // Material indexs
   enum { PointMatl, ShaftMatl, HeadMatl, ResizeMatl };

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Vector direction;
   double length;
};

} // End namespace Widgets
} // End namespace PSECore

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif

//
// $Log$
// Revision 1.3.2.4  2000/11/01 23:02:59  mcole
// Fix for previous merge from trunk
//
// Revision 1.3.2.2  2000/10/26 14:16:56  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.5  2000/06/27 07:57:48  samsonov
// Added Get/SetLength member function
//
// Revision 1.4  2000/06/22 22:39:48  samsonov
// Added resizing mode
// Added rotational functionality in respect to base point
//
// Revision 1.3  1999/10/07 02:07:23  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:38:27  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:05  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:22  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//


#endif
