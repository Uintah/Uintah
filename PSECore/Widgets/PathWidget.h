
/*
 *  PathWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Path_Widget_h
#define SCI_project_Path_Widget_h 1

#include <Widgets/BaseWidget.h>

namespace PSECommon {
namespace Widgets {

using SCICore::GeomSpace::GeomGroup;

class PathPoint;

class PathWidget : public BaseWidget {
   friend class PathPoint;
public:
   PathWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	       Index num_points=10 );
   PathWidget( const PathWidget& );
   virtual ~PathWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   Index GetNumPoints() const;
   
protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   RealVariable* dist;
   RealVariable* hypo;
   
   GeomGroup* pointgroup;
   GeomGroup* tangentgroup;
   GeomGroup* orientgroup;
   GeomGroup* upgroup;
   GeomGroup* splinegroup;

   Index npoints;
   Array1<PathPoint*> points;

   void GenerateSpline();
};

} // End namespace Widgets
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:56:07  mcq
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
