
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

   void Get( Point& p, Vector& tangent, Vector& orient, Vector& up ) const;
   
   void set_scale( const Real scale );

   void SetIndex( const Index i );
   Index GetIndex() const;

private:
   PathWidget* widget;
   Index index;
   
   PointVariable PointVar;
   PointVariable TangentVar;
   PointVariable OrientVar;
   PointVariable UpVar;
   DistanceConstraint ConstTangent;
   DistanceConstraint ConstRight;
   DistanceConstraint ConstOrient;
   DistanceConstraint ConstUp;
   GeomSphere GeomPoint;
   GeomCylinder GeomTangentShaft;
   GeomCappedCone GeomTangentHead;
   GeomCylinder GeomOrientShaft;
   GeomCappedCone GeomOrientHead;
   GeomCylinder GeomUpShaft;
   GeomCappedCone GeomUpHead;
   GeomMaterial PointMatl;
   GeomMaterial TangentShaftMatl;
   GeomMaterial TangentHeadMatl;
   GeomMaterial OrientShaftMatl;
   GeomMaterial OrientHeadMatl;
   GeomMaterial UpShaftMatl;
   GeomMaterial UpHeadMatl;
   GeomGroup* tangent;
   GeomGroup* orient;
   GeomGroup* up;
   GeomPick PickPoint;
   GeomPick PickTangent;
   GeomPick PickOrient;
   GeomPick PickUp;
};

PathPoint::PathPoint( PathWidget* w, const Index i, const Point& p )
: PointVar("Point", w->solve, Scheme1, p),
  TangentVar("Tangent", w->solve, Scheme1, p+Vector(w->dist->real(),0,0)),
  OrientVar("Orient", w->solve, Scheme1, p+Vector(0,w->dist->real(),0)),
  UpVar("Up", w->solve, Scheme2, p+Vector(0,0,w->dist->real())),
  ConstTangent("ConstTangent", 2, &TangentVar, &PointVar, w->dist),
  ConstRight("ConstRight", 2, &OrientVar, &UpVar, w->sqrt2dist),
  ConstOrient("ConstOrient", 2, &OrientVar, &PointVar, w->dist),
  ConstUp("ConstUp", 2, &UpVar, &PointVar, w->dist),
  PointMatl((GeomObj*)&GeomPoint, w->PointMaterial),
  TangentShaftMatl((GeomObj*)&GeomTangentShaft, w->EdgeMaterial),
  TangentHeadMatl((GeomObj*)&GeomTangentHead, w->EdgeMaterial),
  OrientShaftMatl((GeomObj*)&GeomOrientShaft, w->EdgeMaterial),
  OrientHeadMatl((GeomObj*)&GeomOrientHead, w->SpecialMaterial),
  UpShaftMatl((GeomObj*)&GeomUpShaft, w->SpecialMaterial),
  UpHeadMatl((GeomObj*)&GeomUpHead, w->SpecialMaterial),
  PickPoint(&PointMatl, w->module, w, i),
  tangent(new GeomGroup(0)),
  orient(new GeomGroup(0)),
  up(new GeomGroup(0)),
  PickTangent(tangent, w->module, w, i+10000),
  PickOrient(orient, w->module, w, i+20000),
  PickUp(up, w->module, w, i+30000),
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
   ConstUp.VarChoices(Scheme1, 0, 0, 0);
   ConstUp.VarChoices(Scheme2, 0, 0, 0);
   ConstUp.Priorities(P_Default, P_Default, P_Default);

   PointVar.Order();
   TangentVar.Order();
   OrientVar.Order();
   UpVar.Order();
   
   tangent->add(&TangentShaftMatl);
   tangent->add(&TangentHeadMatl);
   orient->add(&OrientShaftMatl);
   orient->add(&OrientHeadMatl);
   up->add(&UpShaftMatl);
   up->add(&UpHeadMatl);
   
   PickPoint.set_highlight(w->HighlightMaterial);
   PickTangent.set_highlight(w->HighlightMaterial);
   PickOrient.set_highlight(w->HighlightMaterial);
   PickUp.set_highlight(w->HighlightMaterial);
   
   w->pointgroup->add(&PickPoint);
   w->tangentgroup->add(&PickTangent);
   w->orientgroup->add(&PickOrient);
   w->upgroup->add(&PickUp);
   w->points.insert(i, this);
   w->npoints++;
}

PathPoint::~PathPoint()
{
   widget->pointgroup->remove(&PickPoint);
   widget->tangentgroup->remove(&PickTangent);
   widget->orientgroup->remove(&PickOrient);
   widget->upgroup->remove(&PickUp);
   widget->points.remove(index);
   widget->npoints--;
}

void
PathPoint::SetIndex( const Index i )
{
   index = i;
   PickPoint.set_widget_data(i);
   PickTangent.set_widget_data(i+10000);
   PickOrient.set_widget_data(i+20000);
   PickUp.set_widget_data(i+30000);
}

void
PathPoint::execute()
{
   Vector v1(((Point)TangentVar-PointVar).normal()),
      v2(((Point)OrientVar-PointVar).normal()),
      v3(((Point)UpVar-PointVar).normal());
   Real shaftlen(3.0*widget->widget_scale), arrowlen(5.0*widget->widget_scale);
   Real spherediam(widget->widget_scale), shaftdiam(0.5*widget->widget_scale);

   if (widget->mode_switches[1]->get_state()) {
      GeomPoint.move(PointVar, spherediam);
   }
   
   if (widget->mode_switches[2]->get_state()) {
      GeomTangentShaft.move(PointVar, (Point)PointVar + v1 * shaftlen, shaftdiam);
      GeomTangentHead.move((Point)PointVar + v1 * shaftlen, (Point)PointVar + v1 * arrowlen, spherediam, 0);
   }
   
   if (widget->mode_switches[3]->get_state()) {
      GeomOrientShaft.move(PointVar, (Point)PointVar + v2 * shaftlen, shaftdiam);
      GeomOrientHead.move((Point)PointVar + v2 * shaftlen, (Point)PointVar + v2 * arrowlen, spherediam, 0);
   }
   
   if (widget->mode_switches[4]->get_state()) {
      GeomUpShaft.move(PointVar, (Point)PointVar + v3 * shaftlen, shaftdiam);
      GeomUpHead.move((Point)PointVar + v3 * shaftlen, (Point)PointVar + v3 * arrowlen, spherediam, 0);
   }

   Vector v(Cross(v2,v3)), v11, v12;
   v1.find_orthogonal(v11,v12);
   PickPoint.set_principal(v1, v11, v12);
   PickTangent.set_principal(v1, v11, v12);
   PickOrient.set_principal(v, v3);
   PickUp.set_principal(v, v3);
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
   case 3:
      UpVar.SetDelta(delta);
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
   UpVar.MoveDelta(delta);
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
PathPoint::Get( Point& p, Vector& tangent, Vector& orient, Vector& up ) const
{
   p = PointVar;
   tangent = ((Point)TangentVar-PointVar).normal();
   orient = ((Point)OrientVar-PointVar).normal();
   up = ((Point)UpVar-PointVar).normal();
}


const Index NumMdes = 5;
const Index NumSwtchs = 5;

PathWidget::PathWidget( Module* module, CrowdMonitor* lock, double widget_scale,
			Index num_points )
: BaseWidget(module, lock, 0, 0, 0, 0, NumMdes, NumSwtchs, widget_scale),
  points(num_points*2), npoints(0)
{
   dist = new RealVariable("Dist", solve, Scheme1, widget_scale*5.0);
   sqrt2dist = new RealVariable("sqrt2Dist", solve, Scheme1, sqrt(2)*widget_scale*5.0);

   splinegroup = new GeomGroup;
   GeomPick* sp = new GeomPick(splinegroup, module, this, -1);
   sp->set_highlight(HighlightMaterial);
   CreateModeSwitch(0, sp);
   pointgroup = new GeomGroup(0);
   CreateModeSwitch(1, pointgroup);
   tangentgroup = new GeomGroup(0);
   CreateModeSwitch(2, tangentgroup);
   orientgroup = new GeomGroup(0);
   CreateModeSwitch(3, orientgroup);
   upgroup = new GeomGroup(0);
   CreateModeSwitch(4, upgroup);

   SetMode(Mode0, Switch0|Switch1|Switch2|Switch3|Switch4);
   SetMode(Mode1, Switch0|Switch1|Switch2);
   SetMode(Mode2, Switch0|Switch1|Switch3|Switch4);
   SetMode(Mode3, Switch0|Switch1);
   SetMode(Mode4, Switch0);

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

   if (mode_switches[0]->get_state()) GenerateSpline();

   for (Index i=0; i<npoints; i++) {
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
			int i )
{
   if (i == -1) // Spline pick.
      MoveDelta(delta);
   else if (i < 10000)
      points[i]->geom_moved(delta, 0);
   else if (i < 20000)
      points[i-10000]->geom_moved(delta, 1);
   else if (i < 30000)
      points[i-20000]->geom_moved(delta, 2);
   else
      points[i-30000]->geom_moved(delta, 3);
   execute();
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

