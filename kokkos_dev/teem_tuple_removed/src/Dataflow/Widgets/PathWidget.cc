/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  PathWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Widgets/PathWidget.h>
#include <Dataflow/Constraints/DistanceConstraint.h>
#include <Dataflow/Constraints/RatioConstraint.h>
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <iostream>
using std::cerr;

namespace SCIRun {


class PathPoint {
public:
  PathPoint( PathWidget* w, const Index i, const Point& p );
  ~PathPoint();

  void execute();
  void geom_moved( const Vector& delta, Index cbdata, const BState& );

  void MoveDelta( const Vector& delta );
  Point ReferencePoint() const;

  void Get( Point& p, Vector& tangent, Vector& orient, Vector& up ) const;
   
  void set_scale( const double scale );

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

// Scheme3 is used by Dist.

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
PathPoint::PathPoint( PathWidget* w, const Index i, const Point& p )
  : widget(w), index(i),
    PointVar("Point", w->solve, Scheme1, p),
    TangentVar("Tangent", w->solve, Scheme1, p+Vector(w->dist->real(),0,0)),
    OrientVar("Orient", w->solve, Scheme1, p+Vector(0,w->dist->real(),0)),
    UpVar("Up", w->solve, Scheme2, p+Vector(0,0,w->dist->real())),
    ConstTangent("ConstTangent", 3, &TangentVar, &PointVar, w->dist),
    ConstRight("ConstRight", 3, &OrientVar, &UpVar, w->hypo),
    ConstOrient("ConstOrient", 3, &OrientVar, &PointVar, w->dist),
    ConstUp("ConstUp", 3, &UpVar, &PointVar, w->dist),
    PointMatl((GeomObj*)&GeomPoint, w->DefaultPointMaterial),
    TangentShaftMatl((GeomObj*)&GeomTangentShaft, w->DefaultEdgeMaterial),
    TangentHeadMatl((GeomObj*)&GeomTangentHead, w->DefaultEdgeMaterial),
    OrientShaftMatl((GeomObj*)&GeomOrientShaft, w->DefaultEdgeMaterial),
    OrientHeadMatl((GeomObj*)&GeomOrientHead, w->DefaultSpecialMaterial),
    UpShaftMatl((GeomObj*)&GeomUpShaft, w->DefaultSpecialMaterial),
    UpHeadMatl((GeomObj*)&GeomUpHead, w->DefaultSpecialMaterial),
    tangent(scinew GeomGroup()),
    orient(scinew GeomGroup()),
    up(scinew GeomGroup()),
    PickPoint(&PointMatl, w->module_, w, i),
    PickTangent(tangent, w->module_, w, i+10000),
    PickOrient(orient, w->module_, w, i+20000),
    PickUp(up, w->module_, w, i+30000)
{
  ConstTangent.VarChoices(Scheme1, 0, 0, 0);
  ConstTangent.VarChoices(Scheme2, 0, 0, 0);
  ConstTangent.VarChoices(Scheme3, 0, 0, 0);
  ConstTangent.Priorities(P_Default, P_Default, P_Default);
  ConstRight.VarChoices(Scheme1, 1, 1, 1);
  ConstRight.VarChoices(Scheme2, 0, 0, 0);
  ConstRight.VarChoices(Scheme3, 1, 0, 1);
  ConstRight.Priorities(P_LowMedium, P_LowMedium, P_LowMedium);
  ConstOrient.VarChoices(Scheme1, 0, 0, 0);
  ConstOrient.VarChoices(Scheme2, 0, 0, 0);
  ConstOrient.VarChoices(Scheme3, 0, 0, 0);
  ConstOrient.Priorities(P_Default, P_Default, P_Default);
  ConstUp.VarChoices(Scheme1, 0, 0, 0);
  ConstUp.VarChoices(Scheme2, 0, 0, 0);
  ConstUp.VarChoices(Scheme3, 0, 0, 0);
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
   
  PickPoint.set_highlight(w->DefaultHighlightMaterial);
  PickTangent.set_highlight(w->DefaultHighlightMaterial);
  PickOrient.set_highlight(w->DefaultHighlightMaterial);
  PickUp.set_highlight(w->DefaultHighlightMaterial);
   
  w->pointgroup->add(&PickPoint);
  w->tangentgroup->add(&PickTangent);
  w->orientgroup->add(&PickOrient);
  w->upgroup->add(&PickUp);
  w->points.insert(w->points.begin() + i, (PathPoint*)this);
}

PathPoint::~PathPoint()
{
  widget->pointgroup->remove(&PickPoint);
  widget->tangentgroup->remove(&PickTangent);
  widget->orientgroup->remove(&PickOrient);
  widget->upgroup->remove(&PickUp);
  widget->points.erase(widget->points.begin() + index);
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
  const Vector v1(((Point)TangentVar-PointVar).normal());
  const Vector v2(((Point)OrientVar-PointVar).normal());
  const Vector v3(((Point)UpVar-PointVar).normal());
  const double shaftlen(3.0*widget->widget_scale_);
  const double arrowlen(5.0*widget->widget_scale_);
  const double sphererad(widget->widget_scale_);
  const double shaftrad(0.5*widget->widget_scale_);

  if (widget->mode_switches[1]->get_state())
  {
    GeomPoint.move(PointVar, sphererad);
  }
   
  if (widget->mode_switches[2]->get_state())
  {
    GeomTangentShaft.move(PointVar, (Point)PointVar + v1 * shaftlen, shaftrad);
    GeomTangentHead.move((Point)PointVar + v1 * shaftlen, (Point)PointVar + v1 * arrowlen, sphererad, 0);
  }
   
  if (widget->mode_switches[3]->get_state())
  {
    GeomOrientShaft.move(PointVar, (Point)PointVar + v2 * shaftlen, shaftrad);
    GeomOrientHead.move((Point)PointVar + v2 * shaftlen, (Point)PointVar + v2 * arrowlen, sphererad, 0);
  }
   
  if (widget->mode_switches[4]->get_state())
  {
    GeomUpShaft.move(PointVar, (Point)PointVar + v3 * shaftlen, shaftrad);
    GeomUpHead.move((Point)PointVar + v3 * shaftlen,
		    (Point)PointVar + v3 * arrowlen, sphererad, 0);
  }

  Vector v(Cross(v2,v3)), v11, v12;
  v1.find_orthogonal(v11,v12);
  PickPoint.set_principal(v1, v11, v12);
  PickTangent.set_principal(v1, v11, v12);
  PickOrient.set_principal(v, v3);
  PickUp.set_principal(v, v3);
}

void
PathPoint::geom_moved( const Vector& delta, Index pick, const BState& )
{
  switch(pick)
  {
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
    cerr << "Unknown case in PathPoint::geom_moved\n";
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


const Index NumCons = 1;
const Index NumVars = 3;
const Index NumMdes = 5;
const Index NumSwtchs = 5;

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
PathWidget::PathWidget( Module* module, CrowdMonitor* lock, double widget_scale,
			Index num_points )
  : BaseWidget(module, lock, "PathWidget", NumVars, NumCons, 0, 0, 0, NumMdes, NumSwtchs, widget_scale),
    points(num_points)
{
  dist = variables[0] = scinew RealVariable("Dist", solve, Scheme3, widget_scale_*5.0);
  hypo = variables[1] = scinew RealVariable("hypo", solve, Scheme3, sqrt(2.0)*widget_scale_*5.0);
  variables[2] = scinew RealVariable("sqrt2", solve, Scheme3, sqrt(2.0));

  constraints[0] = scinew RatioConstraint("ConstSqrt2", 3, hypo, dist, variables[2]);
  constraints[0]->VarChoices(Scheme1, 0, 0, 0);
  constraints[0]->VarChoices(Scheme2, 0, 0, 0);
  constraints[0]->VarChoices(Scheme3, 0, 0, 0);
  constraints[0]->Priorities(P_Highest, P_Highest, P_Highest);

  splinegroup = scinew GeomGroup;
  GeomPick* sp = scinew GeomPick(splinegroup, module, this, -1);
  sp->set_highlight(DefaultHighlightMaterial);
  CreateModeSwitch(0, sp);
  pointgroup = scinew GeomGroup();
  CreateModeSwitch(1, pointgroup);
  tangentgroup = scinew GeomGroup();
  CreateModeSwitch(2, tangentgroup);
  orientgroup = scinew GeomGroup();
  CreateModeSwitch(3, orientgroup);
  upgroup = scinew GeomGroup();
  CreateModeSwitch(4, upgroup);

  SetMode(Mode0, Switch0|Switch1|Switch2|Switch3|Switch4);
  SetMode(Mode1, Switch0|Switch1|Switch2);
  SetMode(Mode2, Switch0|Switch1|Switch3|Switch4);
  SetMode(Mode3, Switch0|Switch1);
  SetMode(Mode4, Switch0);

  double xoffset(2.0*widget_scale_*num_points/2.0);
  for (Index i=0; i<num_points; i++)
  {
    scinew PathPoint(this, i, Point(2.0*widget_scale_*i-xoffset, 0, 0));
  }

  FinishWidget();

  GenerateSpline();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
PathWidget::~PathWidget()
{
  for (Index i=0; i<points.size(); i++)
  {
    delete points[i];
  }
  points.clear();
}


/***************************************************************************
 * The widget's redraw method changes widget geometry to reflect the
 *      widget's variable values and its widget_scale.
 * Geometry should only be changed if the mode_switch that displays
 *      that geometry is active.
 * Redraw should also set the principal directions for all picks.
 * Redraw should never be called directly; the BaseWidget execute method
 *      calls redraw after establishing the appropriate locks.
 */
void
PathWidget::redraw()
{
  dist->Set(widget_scale_*5.0);  // This triggers a LOT of constraints!

  if (mode_switches[0]->get_state()) GenerateSpline();

  for (Index i=0; i<points.size(); i++)
  {
    points[i]->execute();
  }
}


void
PathWidget::GenerateSpline()
{
  splinegroup->remove_all();
  for (Index i=1; i<points.size(); i++)
  {
    splinegroup->add(scinew GeomCylinder(points[i-1]->ReferencePoint(),
					 points[i]->ReferencePoint(),
					 0.33*widget_scale_));
  }
}


/***************************************************************************
 * The widget's geom_moved method receives geometry move requests from
 *      the widget's picks.  The widget's variables must be altered to
 *      reflect these changes based upon which pick made the request.
 * No more than one variable should be Set since this triggers solution of
 *      the constraints--multiple Sets could lead to inconsistencies.
 *      The constraint system only requires that a variable be Set if the
 *      change would cause a constraint to be invalid.  For example, if
 *      all PointVariables are moved by the same delta, then no Set is
 *      required.
 * The last line of the widget's geom_moved method should call the
 *      BaseWidget execute method (which calls the redraw method).
 */
void
PathWidget::geom_moved( GeomPickHandle, int /* axis */, double /* dist */,
			const Vector& delta, int pick, const BState& state,
			const Vector &/*pick_offset*/)
{
  if (pick == -1) // Spline pick.
  {
    MoveDelta(delta);
  }
  else if (pick < 10000)
  {
    points[pick]->geom_moved(delta, 0, state);
  }
  else if (pick < 20000)
  {
    points[pick-10000]->geom_moved(delta, 1, state);
  }
  else if (pick < 30000)
  {
    points[pick-20000]->geom_moved(delta, 2, state);
  }
  else
  {
    points[pick-30000]->geom_moved(delta, 3, state);
  }
  execute(0);
}


/***************************************************************************
 * This standard method simply moves all the widget's PointVariables by
 *      the same delta.
 * The last line of this method should call the BaseWidget execute method
 *      (which calls the redraw method).
 */
void
PathWidget::MoveDelta( const Vector& delta )
{
  for (Index i=0; i<points.size(); i++)
  {
    points[i]->MoveDelta(delta);
  }
  execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
PathWidget::ReferencePoint() const
{
  ASSERT(points.size()>0);
  return points[0]->ReferencePoint();
}


Index
PathWidget::GetNumPoints() const
{
  return points.size();
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
PathWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());
   
  switch(mindex)
  {
  default:
    return "UnknownMaterial";
  }
}

} // End namespace SCIRun

