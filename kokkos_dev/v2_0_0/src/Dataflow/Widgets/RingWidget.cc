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
 *  RingWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Dataflow/Widgets/RingWidget.h>
#include <Dataflow/Constraints/AngleConstraint.h>
#include <Dataflow/Constraints/DistanceConstraint.h>
#include <Dataflow/Constraints/RatioConstraint.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomTorus.h>
#include <Core/Geometry/Plane.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/Trig.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {



const Index NumCons = 6;
const Index NumVars = 8;
const Index NumGeoms = 10;
const Index NumPcks = 10;
const Index NumMatls = 5;
const Index NumMdes = 4;
const Index NumSwtchs = 3;
const Index NumSchemes = 5;

enum { ConstRD, ConstRC, ConstDC, ConstPyth, ConstRadius, ConstAngle };
enum { GeomPointU, GeomPointR, GeomPointD, GeomPointL,
       GeomSlider, GeomRing,
       GeomResizeU, GeomResizeR, GeomResizeD, GeomResizeL };
enum { PickSphU, PickSphR, PickSphD, PickSphL,
       PickRing, PickSlider,
       PickResizeU, PickResizeR, PickResizeD, PickResizeL };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
RingWidget::RingWidget( Module* module, CrowdMonitor* lock, double widget_scale )
  : BaseWidget(module, lock, "RingWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
    oldrightaxis(1, 0, 0), olddownaxis(0, 1, 0)
{
  const double INIT = 5.0*widget_scale;
  // Scheme4/5 are used to resize.
  variables[PointRVar] = scinew PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
  variables[PointDVar] = scinew PointVariable("PntD", solve, Scheme2, Point(0, INIT, 0));
  variables[CenterVar] = scinew PointVariable("Center", solve, Scheme1, Point(0, 0, 0));
  variables[SliderVar] = scinew PointVariable("Slider", solve, Scheme3, Point(INIT, 0, 0));
  variables[DistVar] = scinew RealVariable("Dist", solve, Scheme1, INIT);
  variables[HypoVar] = scinew RealVariable("HYPO", solve, Scheme1, sqrt(2.0*INIT*INIT));
  variables[Sqrt2Var] = scinew RealVariable("Sqrt2", solve, Scheme1, sqrt(2.0));
  variables[AngleVar] = scinew RealVariable("Angle", solve, Scheme3, 0);

  constraints[ConstAngle] = scinew AngleConstraint("ConstAngle",
						   NumSchemes,
						   variables[CenterVar],
						   variables[PointRVar],
						   variables[PointDVar],
						   variables[SliderVar],
						   variables[AngleVar]);
  constraints[ConstAngle]->VarChoices(Scheme1, 3, 3, 3, 3, 3);
  constraints[ConstAngle]->VarChoices(Scheme2, 3, 3, 3, 3, 3);
  constraints[ConstAngle]->VarChoices(Scheme3, 4, 4, 4, 4, 4);
  constraints[ConstAngle]->VarChoices(Scheme4, 3, 3, 3, 3, 3);
  constraints[ConstAngle]->VarChoices(Scheme5, 3, 3, 3, 3, 3);
  constraints[ConstAngle]->Priorities(P_Lowest, P_Lowest, P_Lowest,
				      P_Highest, P_Highest);
  constraints[ConstRadius] = scinew DistanceConstraint("ConstRadius",
						       NumSchemes,
						       variables[CenterVar],
						       variables[SliderVar],
						       variables[DistVar]);
  constraints[ConstRadius]->VarChoices(Scheme1, 1, 1, 1);
  constraints[ConstRadius]->VarChoices(Scheme2, 1, 1, 1);
  constraints[ConstRadius]->VarChoices(Scheme3, 1, 1, 1);
  constraints[ConstRadius]->VarChoices(Scheme4, 1, 1, 1);
  constraints[ConstRadius]->VarChoices(Scheme5, 1, 1, 1);
  constraints[ConstRadius]->Priorities(P_Lowest, P_HighMedium, P_Lowest);
  constraints[ConstRD] = scinew DistanceConstraint("ConstRD",
						   NumSchemes,
						   variables[PointRVar],
						   variables[PointDVar],
						   variables[HypoVar]);
  constraints[ConstRD]->VarChoices(Scheme1, 1, 1, 1);
  constraints[ConstRD]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstRD]->VarChoices(Scheme3, 1, 0, 1);
  constraints[ConstRD]->VarChoices(Scheme4, 1, 1, 1);
  constraints[ConstRD]->VarChoices(Scheme5, 0, 0, 0);
  constraints[ConstRD]->Priorities(P_Default, P_Default, P_Default);
  constraints[ConstPyth] = scinew RatioConstraint("ConstPyth",
						  NumSchemes,
						  variables[HypoVar],
						  variables[DistVar],
						  variables[Sqrt2Var]);
  constraints[ConstPyth]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstPyth]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstPyth]->VarChoices(Scheme3, 0, 0, 0);
  constraints[ConstPyth]->VarChoices(Scheme4, 1, 0, 0);
  constraints[ConstPyth]->VarChoices(Scheme5, 1, 0, 0);
  constraints[ConstPyth]->Priorities(P_Highest, P_Highest, P_Highest);
  constraints[ConstRC] = scinew DistanceConstraint("ConstRC",
						   NumSchemes,
						   variables[PointRVar],
						   variables[CenterVar],
						   variables[DistVar]);
  constraints[ConstRC]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstRC]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstRC]->VarChoices(Scheme3, 0, 0, 0);
  constraints[ConstRC]->VarChoices(Scheme4, 2, 2, 2);
  constraints[ConstRC]->VarChoices(Scheme5, 0, 0, 0);
  constraints[ConstRC]->Priorities(P_Highest, P_Highest, P_Default);
  constraints[ConstDC] = scinew DistanceConstraint("ConstDC",
						   NumSchemes,
						   variables[PointDVar],
						   variables[CenterVar],
						   variables[DistVar]);
  constraints[ConstDC]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstDC]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstDC]->VarChoices(Scheme3, 0, 0, 0);
  constraints[ConstDC]->VarChoices(Scheme4, 0, 0, 0);
  constraints[ConstDC]->VarChoices(Scheme5, 2, 2, 2);
  constraints[ConstDC]->Priorities(P_Highest, P_Highest, P_Default);

  Index geom, pick;
  geometries[GeomRing] = scinew GeomTorus;
  picks_[PickRing] = scinew GeomPick(geometries[GeomRing], module, this, PickRing);
  picks(PickRing)->set_highlight(DefaultHighlightMaterial);
  materials[RingMatl] = scinew GeomMaterial(picks_[PickRing], DefaultEdgeMaterial);
  CreateModeSwitch(0, materials[RingMatl]);

  GeomGroup* pts = scinew GeomGroup;
  for (geom = GeomPointU, pick = PickSphU;
       geom <= GeomPointL; geom++, pick++)
  {
    geometries[geom] = scinew GeomSphere;
    picks_[pick] = scinew GeomPick(geometries[geom], module, this, pick);
    picks(pick)->set_highlight(DefaultHighlightMaterial);
    pts->add(picks_[pick]);
  }
  materials[PointMatl] = scinew GeomMaterial(pts, DefaultPointMaterial);
   
  geometries[GeomResizeL] = scinew GeomCappedCylinder;
  picks_[PickResizeL] = scinew GeomPick(geometries[GeomResizeL],
				       module, this, PickResizeL);
  picks(PickResizeL)->set_highlight(DefaultHighlightMaterial);
  materials[HalfResizeMatl] = scinew GeomMaterial(picks_[PickResizeL], DefaultSpecialMaterial);

  GeomGroup* resulr = scinew GeomGroup;
  geometries[GeomResizeU] = scinew GeomCappedCylinder;
  picks_[PickResizeU] = scinew GeomPick(geometries[GeomResizeU],
				       module, this, PickResizeU);
  picks(PickResizeU)->set_highlight(DefaultHighlightMaterial);
  resulr->add(picks_[PickResizeU]);
  geometries[GeomResizeD] = scinew GeomCappedCylinder;
  picks_[PickResizeD] = scinew GeomPick(geometries[GeomResizeD],
				       module, this, PickResizeD);
  picks(PickResizeD)->set_highlight(DefaultHighlightMaterial);
  resulr->add(picks_[PickResizeD]);
  geometries[GeomResizeR] = scinew GeomCappedCylinder;
  picks_[PickResizeR] = scinew GeomPick(geometries[GeomResizeR],
				       module, this, PickResizeR);
  picks(PickResizeR)->set_highlight(DefaultHighlightMaterial);
  resulr->add(picks_[PickResizeR]);
  materials[ResizeMatl] = scinew GeomMaterial(resulr, DefaultResizeMaterial);

  GeomGroup* w = scinew GeomGroup;
  w->add(materials[PointMatl]);
  w->add(materials[HalfResizeMatl]);
  w->add(materials[ResizeMatl]);
  CreateModeSwitch(1, w);

  geometries[GeomSlider] = scinew GeomCappedCylinder;
  picks_[PickSlider] = scinew GeomPick(geometries[GeomSlider], module, this,
				      PickSlider);
  picks(PickSlider)->set_highlight(DefaultHighlightMaterial);
  materials[SliderMatl] = scinew GeomMaterial(picks_[PickSlider], DefaultSliderMaterial);
  CreateModeSwitch(2, materials[SliderMatl]);

  SetMode(Mode0, Switch0|Switch1|Switch2);
  SetMode(Mode1, Switch0|Switch1);
  SetMode(Mode2, Switch0|Switch2);
  SetMode(Mode3, Switch0);

  FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
RingWidget::~RingWidget()
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
RingWidget::redraw()
{
  Vector Right(GetRightAxis()*variables[DistVar]->real());
  Vector Down(GetDownAxis()*variables[DistVar]->real());
  Point Center(variables[CenterVar]->point());
  Point U(Center-Down);
  Point R(Center+Right);
  Point D(Center+Down);
  Point L(Center-Right);
   
  Vector normal(Cross(GetRightAxis(), GetDownAxis()));
  if (normal.length2() < 1e-6) { normal = Vector(1.0, 0.0, 0.0); }
   
  if (mode_switches[0]->get_state())
  {
    geometry<GeomTorus*>(GeomRing)->move(variables[CenterVar]->point(),
					 normal,
					 variables[DistVar]->real(),
					 0.5*widget_scale_);
  }

  if (mode_switches[1]->get_state())
  {
    geometry<GeomSphere*>(GeomPointU)->move(U, widget_scale_);
    geometry<GeomSphere*>(GeomPointR)->move(R, widget_scale_);
    geometry<GeomSphere*>(GeomPointD)->move(D, widget_scale_);
    geometry<GeomSphere*>(GeomPointL)->move(L, widget_scale_);
    geometry<GeomCappedCylinder*>(GeomResizeU)->move(U, U - (GetDownAxis() * 1.5 * widget_scale_),
						     0.5*widget_scale_);
    geometry<GeomCappedCylinder*>(GeomResizeR)->move(R, R + (GetRightAxis() * 1.5 * widget_scale_),
						     0.5*widget_scale_);
    geometry<GeomCappedCylinder*>(GeomResizeD)->move(D, D + (GetDownAxis() * 1.5 * widget_scale_),
						     0.5*widget_scale_);
    geometry<GeomCappedCylinder*>(GeomResizeL)->move(L, L - (GetRightAxis() * 1.5 * widget_scale_),
						     0.5*widget_scale_);
  }

  Vector slide(variables[SliderVar]->point()-variables[CenterVar]->point());
  if (slide.length2() > 1e-6)
  {
    slide = Cross(normal, slide.normal());
    if (slide.length2() < 1e-6)  { slide = Vector(0.0, 1.0, 0.0); }
  }
  else
  {
    slide = GetRightAxis();
  }
  if (mode_switches[2]->get_state())
  {
    geometry<GeomCappedCylinder*>(GeomSlider)->
      move(variables[SliderVar]->point() - (slide * 0.3 * widget_scale_),
	   variables[SliderVar]->point() + (slide * 0.3 * widget_scale_),
	   1.1*widget_scale_);
  }
   
  ((DistanceConstraint*)constraints[ConstRC])->SetMinimum(1.6*widget_scale_);
  ((DistanceConstraint*)constraints[ConstDC])->SetMinimum(1.6*widget_scale_);
  ((DistanceConstraint*)constraints[ConstRD])->SetMinimum(sqrt(2*1.6*1.6)*widget_scale_);

  Right.normalize();
  Down.normalize();
  Vector Norm(Cross(Right, Down));
  for (Index geom = 0; geom < NumPcks; geom++)
  {
    if (geom == PickSlider)
    {
      picks(geom)->set_principal(slide);
    }
    else if ((geom == PickResizeU) || (geom == PickResizeD))
    {
      picks(geom)->set_principal(Down);
    }
    else if ((geom == PickResizeL) || (geom == PickResizeR))
    {
      picks(geom)->set_principal(Right);
    }
    else if ((geom == PickSphR) || (geom == PickSphL))
    {
      picks(geom)->set_principal(Down, Norm);
    }
    else if ((geom == PickSphU) || (geom == PickSphD))
    {
      picks(geom)->set_principal(Right, Norm);
    }
    else
    {
      picks(geom)->set_principal(Right, Down, Norm);
    }
  }
}


void
RingWidget::geom_pick( GeomPickHandle p,
		       ViewWindow*vw, int data, const BState& bs)
{
  BaseWidget::geom_pick(p, vw, data, bs);
  pick_pointrvar_ = variables[PointRVar]->point();
  pick_pointdvar_ = variables[PointDVar]->point();
  pick_centervar_ = variables[CenterVar]->point();
  pick_slidervar_ = variables[SliderVar]->point();
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
RingWidget::geom_moved( GeomPickHandle, int axis, double dist,
			const Vector& delta, int pick, const BState&,
			const Vector &pick_offset)
{
  Point p;
  const double resize_min(1.5*widget_scale_);
  if (axis==1) dist = -dist;

  ((DistanceConstraint*)constraints[ConstRC])->SetDefault(GetRightAxis());
  ((DistanceConstraint*)constraints[ConstDC])->SetDefault(GetDownAxis());

  switch(pick)
  {
  case PickSphU:
    variables[PointRVar]->Move(pick_pointrvar_);
    variables[PointDVar]->Move(pick_pointdvar_);
    variables[CenterVar]->Move(pick_centervar_);
    variables[SliderVar]->Move(pick_slidervar_);
    variables[PointDVar]->SetDelta(-pick_offset);
    break;

  case PickSphR:
    variables[PointRVar]->Move(pick_pointrvar_);
    variables[PointDVar]->Move(pick_pointdvar_);
    variables[CenterVar]->Move(pick_centervar_);
    variables[SliderVar]->Move(pick_slidervar_);
    variables[PointRVar]->SetDelta(pick_offset);
    break;

  case PickSphD:
    variables[PointRVar]->Move(pick_pointrvar_);
    variables[PointDVar]->Move(pick_pointdvar_);
    variables[CenterVar]->Move(pick_centervar_);
    variables[SliderVar]->Move(pick_slidervar_);
    variables[PointDVar]->SetDelta(pick_offset);
    break;

  case PickSphL:
    variables[PointRVar]->Move(pick_pointrvar_);
    variables[PointDVar]->Move(pick_pointdvar_);
    variables[CenterVar]->Move(pick_centervar_);
    variables[SliderVar]->Move(pick_slidervar_);
    variables[PointRVar]->SetDelta(-pick_offset);
    break;

  case PickResizeU:
    if ((variables[DistVar]->real() - dist) < resize_min)
    {
      p = variables[CenterVar]->point() + GetDownAxis()*resize_min;
    }
    else
    {
      p = variables[PointDVar]->point() - delta;
    }
    variables[PointDVar]->Set(p, Scheme5);
    break;

  case PickResizeR:
    if ((variables[DistVar]->real() + dist) < resize_min)
    {
      p = variables[CenterVar]->point() + GetRightAxis()*resize_min;
    }
    else
    {
      p = variables[PointRVar]->point() + delta;
    }
    variables[PointRVar]->Set(p, Scheme4);
    break;

  case PickResizeD:
    if ((variables[DistVar]->real() + dist) < resize_min)
    {
      p = variables[CenterVar]->point() + GetDownAxis()*resize_min;
    }
    else
    {
      p = variables[PointDVar]->point() + delta;
    }
    variables[PointDVar]->Set(p, Scheme5);
    break;

  case PickResizeL:
    if ((variables[DistVar]->real() - dist) < resize_min)
    {
      p = variables[CenterVar]->point() + GetRightAxis()*resize_min;
    }
    else
    {
      p = variables[PointRVar]->point() - delta;
    }
    variables[PointRVar]->Set(p, Scheme4);
    break;

  case PickSlider:
    variables[SliderVar]->SetDelta(delta);
    break;

  case PickRing:
    variables[PointRVar]->Move(pick_pointrvar_);
    variables[PointDVar]->Move(pick_pointdvar_);
    variables[CenterVar]->Move(pick_centervar_);
    variables[SliderVar]->Move(pick_slidervar_);
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
RingWidget::MoveDelta( const Vector& delta )
{
  variables[PointRVar]->MoveDelta(delta);
  variables[PointDVar]->MoveDelta(delta);
  variables[CenterVar]->MoveDelta(delta);
  variables[SliderVar]->MoveDelta(delta);

  execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
RingWidget::ReferencePoint() const
{
  return variables[CenterVar]->point();
}


void
RingWidget::SetPosition( const Point& center, const Vector& normal, const double radius )
{
  Vector v1, v2;
  normal.find_orthogonal(v1, v2);
  variables[CenterVar]->Move(center);
  variables[PointRVar]->Move(center+v1*radius);
  variables[PointDVar]->Set(center+v2*radius);

  execute(0);
}


void
RingWidget::GetPosition( Point& center, Vector& normal, double& radius ) const
{
  center = variables[CenterVar]->point();
  normal = Plane(variables[PointRVar]->point(),
		 variables[PointDVar]->point(),
		 variables[CenterVar]->point()).normal();
  radius = variables[DistVar]->real();
}


void
RingWidget::SetRatio( const double ratio )
{
  ASSERT((ratio>=0.0) && (ratio<=1.0));
  variables[AngleVar]->Set(ratio*2.0*Pi - Pi);

  execute(0);
}


double
RingWidget::GetRatio() const
{
  double ratio=variables[AngleVar]->real() / (2.0 * Pi);
  if (ratio < 0)
  {
    ratio+=Pi;
  }
  return ratio;
}


void
RingWidget::SetRadius( const double radius )
{
  ASSERT(radius>=0.0);
   
  Vector axis1(variables[PointRVar]->point() - variables[CenterVar]->point());
  Vector axis2(variables[PointDVar]->point() - variables[CenterVar]->point());
  double ratio(radius/variables[DistVar]->real());

  variables[PointRVar]->Move(variables[CenterVar]->point()+axis1*ratio);
  variables[PointDVar]->Move(variables[CenterVar]->point()+axis2*ratio);

  variables[DistVar]->Set(variables[DistVar]->real()*ratio); // This should set the slider...

  execute(0);
}


double
RingWidget::GetRadius() const
{
  return variables[DistVar]->real();
}


const Vector&
RingWidget::GetRightAxis()
{
  Vector axis(variables[PointRVar]->point() - variables[CenterVar]->point());
  if (axis.length2() <= 1e-6)
  {
    return oldrightaxis;
  }
  else
  {
    return (oldrightaxis = axis.normal());
  }
}


const Vector&
RingWidget::GetDownAxis()
{
  Vector axis(variables[PointDVar]->point() - variables[CenterVar]->point());
  if (axis.length2() <= 1e-6)
  {
    return olddownaxis;
  }
  else
  {
    return (olddownaxis = axis.normal());
  }
}


void
RingWidget::GetPlane(Vector& v1, Vector& v2)
{
  v1=GetRightAxis();
  v2=GetDownAxis();
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
RingWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());
   
  switch(mindex)
  {
  case 0:
    return "Point";
  case 1:
    return "Ring";
  case 2:
    return "Slider";
  case 3:
    return "Resize";
  case 4:
    return "HalfResize";
  default:
    return "UnknownMaterial";
  }
}

} // End namespace SCIRun


