
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


class PathPoint;

class PathWidget : public BaseWidget {
   friend class PathPoint;
public:
   PathWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	       Index num_points=10 );
   PathWidget( const PathWidget& );
   virtual ~PathWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   Index GetNumPoints() const;
   
private:
   RealVariable* dist;
   RealVariable* sqrt2dist;
   
   GeomGroup* pointgroup;
   GeomGroup* tangentgroup;
   GeomGroup* orientgroup;
   GeomGroup* upgroup;
   GeomGroup* splinegroup;

   Index npoints;
   Array1<PathPoint*> points;

   void GenerateSpline();
};


#endif
