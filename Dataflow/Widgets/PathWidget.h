
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

#include <Dataflow/Widgets/BaseWidget.h>

namespace SCIRun {


class PathPoint;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

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

} // End namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif


#endif
