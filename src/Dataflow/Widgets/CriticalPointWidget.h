
/*
 *  CriticalPointWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_CriticalPoint_Widget_h
#define SCI_project_CriticalPoint_Widget_h 1

#include <PSECore/Widgets/BaseWidget.h>

namespace PSECore {
namespace Widgets {

class CriticalPointWidget : public BaseWidget {
public:
   // Critical types
   enum CriticalType { Regular, AttractingNode, RepellingNode, Saddle,
		       AttractingFocus, RepellingFocus, SpiralSaddle,
		       NumCriticalTypes };

   CriticalPointWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   CriticalPointWidget( const CriticalPointWidget& );
   virtual ~CriticalPointWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void NextMode();

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetCriticalType( const CriticalType crit );
   Index GetCriticalType() const;

   void SetPosition( const Point& );
   Point GetPosition() const;
   
   void SetDirection( const Vector& v );
   const Vector& GetDirection() const;

   virtual void widget_tcl( TCLArgs& );

   // Variable indexs
   enum { PointVar };

   // Material indexs
   enum { PointMaterial, ShaftMaterial, HeadMaterial, CylinderMatl, TorusMatl, ConeMatl };

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   CriticalType crittype;
   Vector direction;
};

} // End namespace Widgets
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:28  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:06  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:23  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
