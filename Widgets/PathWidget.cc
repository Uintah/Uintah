
/*
 *  PathWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Widgets/PathWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Geom/Cone.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>


class PathPoint {
public:
   PathPoint( PathWidget* w, const Index i, const Point& p );
   ~PathPoint();

   void execute();
   void geom_moved( const Vector& delta, Index cbdata );

   void MoveDelta( const Vector& delta );
   Point ReferencePoint() const;

   void Get( Point& p, Vector& tangent, Vector& orient ) const;
   
   void set_scale( const Real scale );

   void SetIndex( const Index i );
   Index GetIndex() const;

private:
   PathWidget* widget;
   Index index;
   
   PointVariable PointVar;
   PointVariable TangentVar;
   PointVariable OrientVar;
   DistanceConstraint ConstTangent;
   DistanceConstraint ConstRight;
   DistanceConstraint ConstOrient;
   GeomSphere GeomPoint;
   GeomCylinder GeomTangentShaft;
   GeomCappedCone GeomTangentHead;
   GeomCylinder GeomOrientShaft;
   GeomCappedCone GeomOrientHead;
   GeomMaterial PointMatl;
   GeomMaterial TangentShaftMatl;
   GeomMaterial TangentHeadMatl;
   GeomMaterial OrientShaftMatl;
   GeomMaterial OrientHeadMatl;
   GeomGroup tangent;
   GeomGroup orient;
   GeomPick PickPoint;
   GeomPick PickTangent;
   GeomPick PickOrient;
};

PathPoint::PathPoint( PathWidget* w, const Index i, const Point& p )
: PointVar("Point", w->solve, Scheme1, p),
  TangentVar("Tangent", w->solve, Scheme1, p+Vector(w->dist->real(),0,0)),
  OrientVar("Orient", w->solve, Scheme2, p+Vector(0,w->dist->real(),0)),
  ConstTangent("ConstTangent", 2, &TangentVar, &PointVar, w->dist),
  ConstRight("ConstRight", 2, &TangentVar, &OrientVar, w->sqrt2dist),
  ConstOrient("ConstOrient", 2, &OrientVar, &PointVar, w->dist),
  PointMatl((GeomObj*)&GeomPoint, w->PointMaterial),
  TangentShaftMatl((GeomObj*)&GeomTangentShaft, w->EdgeMaterial),
  TangentHeadMatl((GeomObj*)&GeomTangentHead, w->EdgeMaterial),
  OrientShaftMatl((GeomObj*)&GeomOrientShaft, w->EdgeMaterial),
  OrientHeadMatl((GeomObj*)&GeomOrientHead, w->SpecialMaterial),
  tangent(0), orient(0),
  PickPoint(&PointMatl, w->module),
  PickTangent(&tangent, w->module),
  PickOrient(&orient, w->module),
  index(i), widget(w)
{
   ConstTangent.VarChoices(Scheme1, 0, 0, 0);
   ConstTangent.VarChoices(Scheme2, 0, 0, 0);
   ConstTangent.Priorities(P_Default, P_Default, P_Default);
   ConstRight.VarChoices(Scheme1, 1, 1, 1);
   ConstRight.VarChoices(Scheme2, 0, 0, 0);
   ConstRight.Priorities(P_LowMedium, P_LowMedium, P_LowMedium);
   ConstOrient.VarChoices(Scheme1, 0, 0, 0);
   ConstOrient.VarChoices(Scheme2, 0, 0, 0);
   ConstOrient.Priorities(P_Default, P_Default, P_Default);

   PointVar.Order();
   TangentVar.Order();
   OrientVar.Order();
   
   tangent.add(&TangentShaftMatl);
   tangent.add(&TangentHeadMatl);
   orient.add(&OrientShaftMatl);
   orient.add(&OrientHeadMatl);
   
   PickPoint.set_highlight(w->HighlightMaterial);
   PickPoint.set_cbdata((void*)i);
   PickTangent.set_highlight(w->HighlightMaterial);
   PickTangent.set_cbdata((void*)(i+10000));
   PickOrient.set_highlight(w->HighlightMaterial);
   PickOrient.set_cbdata((void*)(i+20000));

   w->pointgroup->add(&PickPoint);
   w->tangentgroup->add(&PickTangent);
   w->orientgroup->add(&PickOrient);
   w->points.insert(i, this);
   w->npoints++;
}

PathPoint::~PathPoint()
{
   widget->pointgroup->remove(&PickPoint);
   widget->tangentgroup->remove(&PickTangent);
   widget->orientgroup->remove(&PickOrient);
   widget->points.remove(index);
   widget->npoints--;
}

void
PathPoint::SetIndex( const Index i )
{
   index = i;
   PickPoint.set_cbdata((void*)(i));
   PickTangent.set_cbdata((void*)(i+10000));
   PickOrient.set_cbdata((void*)(i+20000));
}

void
PathPoint::execute()
{
   Vector v1(((Point)TangentVar-PointVar).normal()), v2(((Point)OrientVar-PointVar).normal());
   
   GeomPoint.move(PointVar, widget->widget_scale);
   GeomTangentShaft.move(PointVar, (Point)PointVar + v1 * widget->widget_scale * 3.0,
			 0.5*widget->widget_scale);
   GeomTangentHead.move((Point)PointVar + v1 * widget->widget_scale * 3.0,
			(Point)PointVar + v1 * widget->widget_scale * 5.0,
			widget->widget_scale, 0);
   GeomOrientShaft.move(PointVar, (Point)PointVar + v2 * widget->widget_scale * 3.0,
			0.5*widget->widget_scale);
   GeomOrientHead.move((Point)PointVar + v2 * widget->widget_scale * 3.0,
		       (Point)PointVar + v2 * widget->widget_scale * 5.0,
		       widget->widget_scale, 0);

   Vector v(Cross(v1,v2));
   PickPoint.set_principal(v, v1, v2);
   PickTangent.set_principal(v, v2);
   PickOrient.set_principal(v, v2);
}

void
PathPoint::geom_moved( const Vector& delta, Index cbdata )
{
   switch(cbdata) {
   case 0:
      MoveDelta(delta);
      break;
   case 1:
      TangentVar.SetDelta(delta);
      break;
   case 2:
      OrientVar.SetDelta(delta);
      break;
   default:
      cerr << "Unknown case in PathPoint::geom_moved" << endl;
      break;
   }
}

void
PathPoint::MoveDelta( const Vector& delta )
{
   PointVar.MoveDelta(delta);
   TangentVar.MoveDelta(delta);
   OrientVar.MoveDelta(delta);
}

Point
PathPoint::ReferencePoint() const
{
   return PointVar.point();
}

Index
PathPoint::GetIndex() const
{
   return index;
}

void
PathPoint::Get( Point& p, Vector& tangent, Vector& orient ) const
{
   p = PointVar;
   tangent = ((Point)TangentVar-PointVar).normal();
   orient = ((Point)OrientVar-PointVar).normal();
}


const Index NumMdes = 5;
const Index NumSwtchs = 4;

PathWidget::PathWidget( Module* module, CrowdMonitor* lock, double widget_scale,
			Index num_points )
: BaseWidget(module, lock, 0, 0, 0, 0, NumMdes, NumSwtchs, widget_scale),
  points(num_points*2), npoints(0)
{
   dist = new RealVariable("Dist", solve, Scheme1, widget_scale*5.0);
   sqrt2dist = new RealVariable("sqrt2Dist", solve, Scheme1, sqrt(2)*widget_scale*5.0);

   splinegroup = new GeomGroup;
   GeomPick* sp = new GeomPick(splinegroup, module);
   sp->set_highlight(HighlightMaterial);
   sp->set_cbdata((void*)-1);
   CreateModeSwitch(0, sp);
   pointgroup = new GeomGroup(0);
   CreateModeSwitch(1, pointgroup);
   tangentgroup = new GeomGroup(0);
   CreateModeSwitch(2, tangentgroup);
   orientgroup = new GeomGroup(0);
   CreateModeSwitch(3, orientgroup);

   SetMode(Mode1, Switch0|Switch1|Switch2|Switch3);
   SetMode(Mode2, Switch0|Switch1|Switch2);
   SetMode(Mode3, Switch0|Switch1|Switch3);
   SetMode(Mode4, Switch0|Switch1);
   SetMode(Mode5, Switch0);

   Real xoffset(2.0*widget_scale*num_points/2.0);
   for (Index i=0; i<num_points; i++)
      new PathPoint(this, i, Point(2.0*widget_scale*i-xoffset, 0, 0));

   FinishWidget();

   GenerateSpline();
}


PathWidget::~PathWidget()
{
   for (Index i=0; i<npoints; i++)
      delete points[i];
   points.remove_all();
   npoints = 0;
}


void
PathWidget::widget_execute()
{
   cout << "Setting dist" << endl;
   dist->Set(widget_scale*5.0);  // This triggers a LOT of constraints!
   cout << "Setting sqrt2dist" << endl;
   sqrt2dist->Set(sqrt(2)*widget_scale*5.0);  // This triggers a LOT of constraints!

   GenerateSpline();

   for (Index i=0; i<npoints; i++) {
      cout << "Executing " << i << endl;
      points[i]->execute();
   }
}


void
PathWidget::GenerateSpline()
{
   splinegroup->remove_all();
   for (Index i=1; i<npoints; i++) {
      splinegroup->add(new GeomCylinder(points[i-1]->ReferencePoint(),
					points[i]->ReferencePoint(),
					0.33*widget_scale));
   }
}


void
PathWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   int i((int)cbdata);

   if (i == -1) // Spline pick.
      MoveDelta(delta);
   else if (i < 10000)
      points[i]->geom_moved(delta, 0);
   else if (i < 20000)
      points[i-10000]->geom_moved(delta, 1);
   else
      points[i-20000]->geom_moved(delta, 2);
}


void
PathWidget::MoveDelta( const Vector& delta )
{
   for (Index i=0; i<npoints; i++)
      points[i]->MoveDelta(delta);
   
   execute();
}


Index
PathWidget::GetNumPoints() const
{
   return npoints;
}

