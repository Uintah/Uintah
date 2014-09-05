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
 *  ArrowWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Dataflow/Widgets/ArrowWidget.h>
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Constraints/DistanceConstraint.h>

namespace SCIRun {



const Index NumCons = 1;
const Index NumVars = 3;
const Index NumGeoms = 4;
const Index NumPcks = 4;
const Index NumMatls = 4;
const Index NumMdes = 3;
const Index NumSwtchs = 4;
const Index NumSchemes = 2;

enum { GeomPoint, GeomShaft, GeomHead, GeomResize };
enum { PointP, ShaftP, HeadP, ResizeP };
enum { ConstDist };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */

ArrowWidget::ArrowWidget( Module* module, CrowdMonitor* lock, double widget_scale )
  : BaseWidget(module, lock, "ArrowWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
    direction(1, 0, 0)
{
  length = 1;
  variables[PointVar] = scinew PointVariable("Point", solve, Scheme1, Point(0, 0, 0));
  variables[HeadVar]  = scinew PointVariable("Head", solve, Scheme1, Point(length, 0, 0));
  variables[DistVar]  = scinew RealVariable("Dist", solve,  Scheme1, length);

  constraints[ConstDist]=scinew DistanceConstraint("ConstDist",
						   NumSchemes,
						   variables[PointVar],
						   variables[HeadVar],
						   variables[DistVar]);
  constraints[ConstDist]->VarChoices(Scheme1, 0, 1, 0);
  constraints[ConstDist]->VarChoices(Scheme2, 2, 2, 2);
  constraints[ConstDist]->Priorities(P_Lowest, P_Lowest, P_Highest);

						
  geometries[GeomPoint] = scinew GeomSphere;
  picks_[PointP] = scinew GeomPick(geometries[GeomPoint], module, this, PointP);
  picks(PointP)->set_highlight(DefaultHighlightMaterial);
  materials[PointMatl] = scinew GeomMaterial(picks_[PointP], DefaultPointMaterial);
  CreateModeSwitch(0, materials[PointMatl]);

  geometries[GeomShaft] = scinew GeomCylinder;
  picks_[ShaftP] = scinew GeomPick(geometries[GeomShaft], module, this, ShaftP);
  picks(ShaftP)->set_highlight(DefaultHighlightMaterial);
  materials[ShaftMatl] = scinew GeomMaterial(picks_[ShaftP], DefaultEdgeMaterial);
  CreateModeSwitch(1, materials[ShaftMatl]);

  geometries[GeomHead] = scinew GeomCappedCone;
  picks_[HeadP] = scinew GeomPick(geometries[GeomHead], module, this, HeadP);
  picks(HeadP)->set_highlight(DefaultHighlightMaterial);
  materials[HeadMatl] = scinew GeomMaterial(picks_[HeadP], DefaultEdgeMaterial);
  CreateModeSwitch(2, materials[HeadMatl]);

  geometries[GeomResize] = scinew GeomCappedCylinder;
  picks_[ResizeP] = scinew GeomPick(geometries[GeomResize], module, this, ResizeP);
  picks(ResizeP)->set_highlight(DefaultHighlightMaterial);
  materials[ResizeMatl] = scinew GeomMaterial(picks_[ResizeP], DefaultResizeMaterial);
  CreateModeSwitch(3, materials[ResizeMatl]);

  SetMode(Mode0, Switch0|Switch1|Switch2);
  SetMode(Mode1, Switch0|Switch1|Switch2|Switch3);
  SetMode(Mode2, Switch0);
  FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
ArrowWidget::~ArrowWidget()
{
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
ArrowWidget::redraw()
{
  Point P(variables[PointVar]->point()), H(variables[HeadVar]->point());

  if (mode_switches[0]->get_state())
  {
    geometry<GeomSphere*>(GeomPoint)->move(P, widget_scale_);
    geometry<GeomCylinder*>(GeomShaft)->move(P, H, 0.5*widget_scale_);
    geometry<GeomCappedCone*>(GeomHead)->move(H, H +GetDirection()*widget_scale_* 2.0, widget_scale_, 0);
  }
  if (mode_switches[1]->get_state())
  {
    geometry<GeomCappedCylinder*>(GeomResize)->move(H+GetDirection()*widget_scale_*1.0,  H + GetDirection()*widget_scale_*1.5, widget_scale_);
  }

  if (mode_switches[2]->get_state())
  {
    geometry<GeomSphere*>(GeomPoint)->move(P, widget_scale_);
  }

  Vector v(GetDirection()), v1, v2;
  v.find_orthogonal(v1,v2);
  for (Index geom = 0; geom < NumPcks; geom++)
  {
    if (geom==ResizeP)
    {
      picks(geom)->set_principal(v);
    }
    else
    {
      picks(geom)->set_principal(v, v1, v2);
    }
  }
}



void
ArrowWidget::geom_pick(GeomPickHandle p,
		       ViewWindow *vw, int data, const BState &bs)
{
  BaseWidget::geom_pick(p, vw, data, bs);
  pick_pointvar_ = variables[PointVar]->point();
  pick_headvar_ = variables[HeadVar]->point();
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
ArrowWidget::geom_moved( GeomPickHandle, int /* axis */, double /* dist */,
			 const Vector& /*delta*/, int pick, const BState&,
			 const Vector &pick_offset)
{
  ((DistanceConstraint*)constraints[ConstDist])->SetDefault(GetDirection());
  switch(pick)
  {
  case HeadP:
    variables[PointVar]->Move(pick_pointvar_);
    variables[HeadVar]->Move(pick_headvar_);
    variables[HeadVar]->SetDelta(pick_offset, Scheme1);
    break;

  case ResizeP:
    variables[PointVar]->Move(pick_pointvar_);
    variables[HeadVar]->Move(pick_headvar_);
    variables[HeadVar]->SetDelta(pick_offset, Scheme2);
    break;

  case PointP:
  case ShaftP:
    variables[PointVar]->Move(pick_pointvar_);
    variables[HeadVar]->Move(pick_headvar_);
    MoveDelta(pick_offset);
    break;
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
ArrowWidget::MoveDelta( const Vector& delta )
{
  variables[PointVar]->MoveDelta(delta);
  variables[HeadVar]->MoveDelta(delta);

  execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
ArrowWidget::ReferencePoint() const
{
  return variables[PointVar]->point();
}


void
ArrowWidget::SetPosition( const Point& p )
{
  variables[HeadVar]->MoveDelta(p-(variables[PointVar]->point()));
  variables[PointVar]->Move(p);
  execute(0);
}


Point
ArrowWidget::GetPosition() const
{
  return variables[PointVar]->point();
}


void
ArrowWidget::SetDirection( const Vector& v )
{

  // variables[Dist]*(v.norm()-direction.norm())
  variables[HeadVar]->MoveDelta((v.normal()-direction.normal())*variables[DistVar]->real());
  //   variables[HeadVar]->SetDelta(v-direction, Scheme1);
  direction = v;
  execute(0);
}

// by AS: updates if nessesary direction and returns it
const Vector&
ArrowWidget::GetDirection()
{
  Vector dir(variables[HeadVar]->point() - variables[PointVar]->point());
  if (dir.length2() <= 1e-6)
  {
    return direction;
  }
  else
  {
    return (direction = dir.normal());
  }
}

void
ArrowWidget::SetLength( double new_length )
{
  Vector delta=GetDirection()*(new_length-GetLength());
  variables[HeadVar]->SetDelta(delta, Scheme2);
  execute(0);
}

double
ArrowWidget::GetLength()
{
  Vector dir(variables[HeadVar]->point() - variables[PointVar]->point());
  return (length=dir.length());
}

/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the
 *      BaseWidget UI.
 */
string
ArrowWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());

  switch(mindex)
  {
  case 0:
    return "Point";
  case 1:
    return "Shaft";
  case 2:
    return "Head";
  default:
    return "UnknownMaterial";
  }
}


void
ArrowWidget::widget_tcl( GuiArgs& args )
{
  if (args[1] == "translate")
  {
    if (args.count() != 4)
    {
      args.error("arrow widget needs axis translation");
      return;
    }
    double trans;
    if (!string_to_double(args[3], trans))
    {
      args.error("arrow widget can't parse translation `" + args[3] + "'");
      return;
    }
    Point p(GetPosition());
    switch (args[2][0])
    {
    case 'x':
      p.x(trans);
      break;
    case 'y':
      p.y(trans);
      break;
    case 'z':
      p.z(trans);
      break;
    default:
      args.error("arrow widget unknown axis `" + args[2] + "'");
      break;
    }
    SetPosition(p);
  }
}

} // End namespace SCIRun
