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
 *  FrameWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Dataflow/Widgets/FrameWidget.h>
#include <Dataflow/Constraints/DistanceConstraint.h>
#include <Dataflow/Constraints/PythagorasConstraint.h>
#include <Dataflow/Constraints/RatioConstraint.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {



const Index NumSwitches = 4;
const Index NumSchemes = 5;

enum { ConstRD, ConstRC, ConstDC, ConstPyth,
       ConstRatioR, ConstRatioD, NumCons };
enum { GeomSPointUL, GeomSPointUR, GeomSPointDR, GeomSPointDL,
       GeomPointU, GeomPointR, GeomPointD, GeomPointL,
       GeomCylU, GeomCylR, GeomCylD, GeomCylL,
       GeomResizeU, GeomResizeR, GeomResizeD, GeomResizeL,
       GeomSliderCylR, GeomSliderCylD, NumGeoms };
enum { PickSphU, PickSphR, PickSphD, PickSphL, PickCyls,
       PickResizeU, PickResizeR, PickResizeD, PickResizeL,
       PickSliderR, PickSliderD, NumPicks };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
FrameWidget::FrameWidget( Module* module, CrowdMonitor* lock,
			  double widget_scale,
			  bool is_slideable ) :
  BaseWidget(module, lock, "FrameWidget", NumVars, NumCons, NumGeoms,
	     NumPicks, NumMatls, 0, NumSwitches, widget_scale),
  is_slideable_(is_slideable),
  oldrightaxis(1, 0, 0),
  olddownaxis(0, 1, 0)
{
  const double INIT = 5.0*widget_scale;
  variables[CenterVar] = scinew PointVariable("Center", solve, Scheme1, Point(0, 0, 0));
  variables[PointRVar] = scinew PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
  variables[PointDVar] = scinew PointVariable("PntD", solve, Scheme2, Point(0, INIT, 0));
  variables[DistRVar] = scinew RealVariable("RDIST", solve, Scheme3, INIT);
  variables[DistDVar] = scinew RealVariable("DDIST", solve, Scheme4, INIT);
  variables[HypoVar] = scinew RealVariable("HYPO", solve, Scheme3, sqrt(2*INIT*INIT));
  variables[SDistRVar] = scinew RealVariable("SDistR", solve, Scheme5, INIT/2.0);
  variables[SDistDVar] = scinew RealVariable("SDistD", solve, Scheme5, INIT/2.0);
  variables[RatioRVar] = scinew RealVariable("RatioR", solve, Scheme1, 0.5);
  variables[RatioDVar] = scinew RealVariable("RatioD", solve, Scheme1, 0.5);

  constraints[ConstRatioR] = scinew RatioConstraint("ConstRatioR",
						    NumSchemes,
						    variables[SDistRVar],
						    variables[DistRVar],
						    variables[RatioRVar]);
  constraints[ConstRatioR]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstRatioR]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstRatioR]->VarChoices(Scheme3, 0, 0, 0);
  constraints[ConstRatioR]->VarChoices(Scheme4, 0, 0, 0);
  constraints[ConstRatioR]->VarChoices(Scheme5, 2, 2, 2);
  constraints[ConstRatioR]->Priorities(P_Highest, P_Highest, P_Highest);
  constraints[ConstRatioD] = scinew RatioConstraint("ConstRatioD",
						    NumSchemes,
						    variables[SDistDVar],
						    variables[DistDVar],
						    variables[RatioDVar]);
  constraints[ConstRatioD]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstRatioD]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstRatioD]->VarChoices(Scheme3, 0, 0, 0);
  constraints[ConstRatioD]->VarChoices(Scheme4, 0, 0, 0);
  constraints[ConstRatioD]->VarChoices(Scheme5, 2, 2, 2);
  constraints[ConstRatioD]->Priorities(P_Highest, P_Highest, P_Highest);
  constraints[ConstRD] = scinew DistanceConstraint("ConstRD",
						   NumSchemes,
						   variables[PointRVar],
						   variables[PointDVar],
						   variables[HypoVar]);
  constraints[ConstRD]->VarChoices(Scheme1, 1, 1, 1);
  constraints[ConstRD]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstRD]->VarChoices(Scheme3, 2, 2, 1);
  constraints[ConstRD]->VarChoices(Scheme4, 2, 2, 0);
  constraints[ConstRD]->VarChoices(Scheme5, 1, 0, 1);
  constraints[ConstRD]->Priorities(P_Default, P_Default, P_Default);
  constraints[ConstPyth] = scinew PythagorasConstraint("ConstPyth",
						       NumSchemes,
						       variables[DistRVar],
						       variables[DistDVar],
						       variables[HypoVar]);
  constraints[ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
  constraints[ConstPyth]->VarChoices(Scheme2, 1, 0, 0);
  constraints[ConstPyth]->VarChoices(Scheme3, 2, 2, 1);
  constraints[ConstPyth]->VarChoices(Scheme4, 2, 2, 0);
  constraints[ConstPyth]->VarChoices(Scheme5, 1, 0, 1);
  constraints[ConstPyth]->Priorities(P_Highest, P_Highest, P_Highest);
  constraints[ConstRC] = scinew DistanceConstraint("ConstRC",
						   NumSchemes,
						   variables[PointRVar],
						   variables[CenterVar],
						   variables[DistRVar]);
  constraints[ConstRC]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstRC]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstRC]->VarChoices(Scheme3, 2, 2, 2);
  constraints[ConstRC]->VarChoices(Scheme4, 0, 0, 0);
  constraints[ConstRC]->VarChoices(Scheme5, 1, 0, 1);
  constraints[ConstRC]->Priorities(P_Highest, P_Highest, P_Default);
  constraints[ConstDC] = scinew DistanceConstraint("ConstDC",
						   NumSchemes,
						   variables[PointDVar],
						   variables[CenterVar],
						   variables[DistDVar]);
  constraints[ConstDC]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstDC]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstDC]->VarChoices(Scheme3, 0, 0, 0);
  constraints[ConstDC]->VarChoices(Scheme4, 2, 2, 2);
  constraints[ConstDC]->VarChoices(Scheme5, 1, 0, 1);
  constraints[ConstDC]->Priorities(P_Highest, P_Highest, P_Default);

  Index geom, pick;
  GeomGroup* cyls = scinew GeomGroup;
  for (geom = GeomSPointUL; geom <= GeomSPointDL; geom++)
  {
    geometries[geom] = scinew GeomSphere;
    cyls->add(geometries[geom]);
  }
  for (geom = GeomCylU; geom <= GeomCylL; geom++)
  {
    geometries[geom] = scinew GeomCylinder;
    cyls->add(geometries[geom]);
  }
  picks_[PickCyls] = scinew GeomPick(cyls, module, this, PickCyls);
  picks(PickCyls)->set_highlight(DefaultHighlightMaterial);
  materials[EdgeMatl] = scinew GeomMaterial(picks_[PickCyls], DefaultEdgeMaterial);
  CreateModeSwitch(0, materials[EdgeMatl]);

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
  CreateModeSwitch(1, materials[PointMatl]);
   
  GeomGroup* resizes = scinew GeomGroup;
  for (geom = GeomResizeU, pick = PickResizeU;
       geom <= GeomResizeL; geom++, pick++)
  {
    geometries[geom] = scinew GeomCappedCylinder;
    picks_[pick] = scinew GeomPick(geometries[geom], module, this, pick);
    picks(pick)->set_highlight(DefaultHighlightMaterial);
    resizes->add(picks_[pick]);
  }
  materials[ResizeMatl] = scinew GeomMaterial(resizes, DefaultResizeMaterial);
  CreateModeSwitch(2, materials[ResizeMatl]);

  GeomGroup* sliders = scinew GeomGroup;
  geometries[GeomSliderCylR] = scinew GeomCappedCylinder;
  picks_[PickSliderR] = scinew GeomPick(geometries[GeomSliderCylR], module, this, PickSliderR);
  picks(PickSliderR)->set_highlight(DefaultHighlightMaterial);
  sliders->add(picks_[PickSliderR]);
  geometries[GeomSliderCylD] = scinew GeomCappedCylinder;
  picks_[PickSliderD] = scinew GeomPick(geometries[GeomSliderCylD], module, this, PickSliderD);
  picks(PickSliderD)->set_highlight(DefaultHighlightMaterial);
  sliders->add(picks_[PickSliderD]);
  materials[SliderMatl] = scinew GeomMaterial(sliders, DefaultSliderMaterial);
  CreateModeSwitch(3, materials[SliderMatl]);

  // Switch0 are the bars
  // Switch1 are the rotation points
  // Switch2 are the resize cylinders
  // Switch3 are the sliders
  if (is_slideable_)
  {
    SetNumModes(7);
    SetMode(Mode0, Switch0|Switch1|Switch2|Switch3);
    SetMode(Mode1, Switch0|Switch1|Switch2);
    SetMode(Mode2, Switch0|Switch3);
    SetMode(Mode3, Switch0);
    SetMode(Mode4, Switch0|Switch1|Switch3);
    SetMode(Mode5, Switch0|Switch2|Switch3);
    SetMode(Mode6, Switch0|Switch2);
  }
  else
  {
    SetNumModes(4);
    SetMode(Mode0, Switch0|Switch1|Switch2);
    SetMode(Mode1, Switch0);
    SetMode(Mode2, Switch0|Switch1);
    SetMode(Mode3, Switch0|Switch2);
  }
   
  FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
FrameWidget::~FrameWidget()
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
FrameWidget::redraw()
{
  const double sphererad(widget_scale_);
  const double resizerad(0.5*widget_scale_);
  const double cylinderrad(0.5*widget_scale_);
  Vector Right(GetRightAxis()*variables[DistRVar]->real());
  Vector Down(GetDownAxis()*variables[DistDVar]->real());
  Point Center(variables[CenterVar]->point());
  Point UL(Center-Right-Down);
  Point UR(Center+Right-Down);
  Point DR(Center+Right+Down);
  Point DL(Center-Right+Down);
  Point U(Center-Down);
  Point R(Center+Right);
  Point D(Center+Down);
  Point L(Center-Right);

  // draw the bars
  if (mode_switches[0]->get_state())
  {
    geometry<GeomCylinder*>(GeomCylU)->move(UL, UR, cylinderrad);
    geometry<GeomCylinder*>(GeomCylR)->move(UR, DR, cylinderrad);
    geometry<GeomCylinder*>(GeomCylD)->move(DR, DL, cylinderrad);
    geometry<GeomCylinder*>(GeomCylL)->move(DL, UL, cylinderrad);
    geometry<GeomSphere*>(GeomSPointUL)->move(UL, cylinderrad);
    geometry<GeomSphere*>(GeomSPointUR)->move(UR, cylinderrad);
    geometry<GeomSphere*>(GeomSPointDR)->move(DR, cylinderrad);
    geometry<GeomSphere*>(GeomSPointDL)->move(DL, cylinderrad);
  }

  // draw the rotation points
  if (mode_switches[1]->get_state())
  {
    geometry<GeomSphere*>(GeomPointU)->move(U, sphererad);
    geometry<GeomSphere*>(GeomPointR)->move(R, sphererad);
    geometry<GeomSphere*>(GeomPointD)->move(D, sphererad);
    geometry<GeomSphere*>(GeomPointL)->move(L, sphererad);
  }

  // draw the resize cylinders
  if (mode_switches[2]->get_state())
  {
    const Vector resizeR(GetRightAxis()*1.5*widget_scale_);
    const Vector resizeD(GetDownAxis()*1.5*widget_scale_);
      
    geometry<GeomCappedCylinder*>(GeomResizeU)->move(U, U - resizeD, resizerad);
    geometry<GeomCappedCylinder*>(GeomResizeR)->move(R, R + resizeR, resizerad);
    geometry<GeomCappedCylinder*>(GeomResizeD)->move(D, D + resizeD, resizerad);
    geometry<GeomCappedCylinder*>(GeomResizeL)->move(L, L - resizeR, resizerad);
  }

  // draw the sliders
  if (mode_switches[3]->get_state())
  {
    Point SliderR(UL+GetRightAxis()*variables[SDistRVar]->real()*2.0);
    Point SliderD(UL+GetDownAxis()*variables[SDistDVar]->real()*2.0);
    geometry<GeomCappedCylinder*>(GeomSliderCylR)->move(SliderR - (GetRightAxis() * 0.3 * widget_scale_),
							SliderR + (GetRightAxis() * 0.3 * widget_scale_),
							1.1*widget_scale_);
    geometry<GeomCappedCylinder*>(GeomSliderCylD)->move(SliderD - (GetDownAxis() * 0.3 * widget_scale_),
							SliderD + (GetDownAxis() * 0.3 * widget_scale_),
							1.1*widget_scale_);
  }

  ((DistanceConstraint*)constraints[ConstRC])->SetMinimum(1.6*widget_scale_);
  ((DistanceConstraint*)constraints[ConstDC])->SetMinimum(1.6*widget_scale_);
  ((DistanceConstraint*)constraints[ConstRD])->SetMinimum(sqrt(2*1.6*1.6)*widget_scale_);

  Right.normalize();
  Down.normalize();
  Vector Norm(Cross(Right, Down));
  for (Index geom = 0; geom < NumPicks; geom++)
  {
    if ((geom == PickResizeU) || (geom == PickResizeD) || (geom == PickSliderD))
    {
      picks(geom)->set_principal(Down);
    }
    else if ((geom == PickResizeL) || (geom == PickResizeR) || (geom == PickSliderR))
    {
      picks(geom)->set_principal(Right);
    }
    else if ((geom == PickSphU) || (geom == PickSphD))
    {
      picks(geom)->set_principal(Right, Norm);
    }
    else if ((geom == PickSphL) || (geom == PickSphR))
    {
      picks(geom)->set_principal(Down, Norm);
    }
    else
    {
      picks(geom)->set_principal(Right, Down, Norm);
    }
  }
}

// if rotating, save the start position of the selected widget 
void
FrameWidget::geom_pick(GeomPickHandle p,
		       ViewWindow*w, int pick, const BState& state)
{
  BaseWidget::geom_pick(p, w, pick, state);

  pick_centervar_ = variables[CenterVar]->point();
  pick_pointdvar_ = variables[PointDVar]->point();
  pick_pointrvar_ = variables[PointRVar]->point();

  const Point c2 = (variables[CenterVar]->point().vector()*2).point();
  Point start_point;
  switch(pick)
  {
  case PickSphU:
    start_point = (c2 - pick_pointdvar_).point();
    break;

  case PickSphD:
    start_point = pick_pointdvar_;
    break;

  case PickSphL:
    start_point =(c2 - pick_pointrvar_).point();
    break;

  case PickSphR:
    start_point = pick_pointrvar_;
    break;

  default:
    return;
  }
  rot_start_ray_ = start_point - pick_centervar_;
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
FrameWidget::geom_moved( GeomPickHandle, int axis, double dist,
			 const Vector& delta, int pick, const BState&,
			 const Vector &pick_offset)
{
  Vector delt(delta);
  double ResizeMin(1.5*widget_scale_);
  if (axis==1) dist = -dist;

  ((DistanceConstraint*)constraints[ConstRC])->SetDefault(GetRightAxis());
  ((DistanceConstraint*)constraints[ConstDC])->SetDefault(GetDownAxis());
   
  Transform trans;
  Vector rot_axis;
  Point c(variables[CenterVar]->point());
  switch(pick)
  {
  case PickSphU: case PickSphD: 
  case PickSphL: case PickSphR:
    {
      Vector rot_start_ray_norm = rot_start_ray_;
      rot_start_ray_norm.normalize();
      Vector rot_current_ray_norm = rot_start_ray_ + pick_offset;
      rot_current_ray_norm.normalize();
      rot_axis=Cross(rot_start_ray_norm, rot_current_ray_norm);
      if (rot_axis.length2()<1.e-16) { rot_axis=Vector(1,0,0); }
      else { rot_axis.normalize(); }

      trans.post_translate(c.vector());
      const double amount = pick_offset.length() / rot_start_ray_.length();
      trans.post_rotate(amount, rot_axis);
      trans.post_translate(-c.vector());
      variables[PointDVar]->Move(trans.project(pick_pointdvar_));
      variables[PointRVar]->Move(trans.project(pick_pointrvar_));
    }
    break;

  case PickResizeU:
    if ((variables[DistDVar]->real() - dist) < ResizeMin)
    {
      delt = variables[CenterVar]->point() + GetDownAxis()*ResizeMin
	- variables[PointDVar]->point();
    }
    variables[PointRVar]->MoveDelta(delt/2.0);      
    variables[CenterVar]->SetDelta(delt/2.0, Scheme4);
    break;

  case PickResizeR:
    if ((variables[DistRVar]->real() + dist) < ResizeMin)
    {
      delt = variables[CenterVar]->point() + GetDownAxis()*ResizeMin
	- variables[PointDVar]->point();
    }
    variables[CenterVar]->MoveDelta(delt/2.0);
    variables[PointDVar]->MoveDelta(delt/2.0);      
    variables[PointRVar]->SetDelta(delt, Scheme3);
    break;

  case PickResizeD:
    if ((variables[DistDVar]->real() + dist) < ResizeMin)
    {
      delt = variables[CenterVar]->point() + GetDownAxis()*ResizeMin
	- variables[PointDVar]->point();
    }
    variables[CenterVar]->MoveDelta(delt/2.0);
    variables[PointRVar]->MoveDelta(delt/2.0);      
    variables[PointDVar]->SetDelta(delt, Scheme4);
    break;

  case PickResizeL:
    if ((variables[DistRVar]->real() - dist) < ResizeMin)
    {
      delt = variables[CenterVar]->point() + GetDownAxis()*ResizeMin 
	- variables[PointDVar]->point();
    }
    variables[PointDVar]->MoveDelta(delt/2.0);      
    variables[CenterVar]->SetDelta(delt/2.0, Scheme3);
    break;

  case PickSliderR:
    if (is_slideable_)
    {
      double sdist(variables[SDistRVar]->real()+dist/2.0);
      if (sdist<0.0) { sdist=0.0; }
      else if (sdist>variables[DistRVar]->real())
      {
	sdist=variables[DistRVar]->real();
      }
      variables[SDistRVar]->Set(sdist);
    }
    break;

  case PickSliderD:
    if (is_slideable_)
    {
      double sdist = variables[SDistDVar]->real()+dist/2.0;
      if (sdist<0.0) { sdist=0.0; }
      else if (sdist>variables[DistDVar]->real())
      {
	sdist=variables[DistDVar]->real();
      }
      variables[SDistDVar]->Set(sdist);
    }
    break;

  case PickCyls:
    variables[CenterVar]->Move(pick_centervar_);
    variables[PointDVar]->Move(pick_pointdvar_);
    variables[PointRVar]->Move(pick_pointrvar_);
    MoveDelta(pick_offset);
    break;
  default:
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
FrameWidget::MoveDelta( const Vector& delta )
{
  variables[PointRVar]->MoveDelta(delta);
  variables[PointDVar]->MoveDelta(delta);
  variables[CenterVar]->MoveDelta(delta);

  execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
FrameWidget::ReferencePoint() const
{
  return variables[CenterVar]->point();
}


void
FrameWidget::SetPosition( const Point& center, const Point& R, const Point& D )
{
  variables[PointRVar]->Move(R);
  variables[PointDVar]->Move(D);
  const double sizeR((R-center).length());
  const double sizeD((D-center).length());
  variables[DistRVar]->Move(sizeR);
  variables[CenterVar]->Move(center);
  variables[DistDVar]->Set(sizeD);
  //   variables[DistDVar]->Move(sizeD);
  //   variables[CenterVar]->Set(center, Scheme3); // This should set Hypo...
  variables[SDistRVar]->Set(sizeR*variables[RatioRVar]->real(), Scheme1); // Slider1...
  variables[SDistDVar]->Set(sizeD*variables[RatioDVar]->real(), Scheme1); // Slider2...

  execute(0);
}


void
FrameWidget::GetPosition( Point& center, Point& R, Point& D )
{
  center = variables[CenterVar]->point();
  R = variables[PointRVar]->point();
  D = variables[PointDVar]->point();
}


void
FrameWidget::SetPosition( const Point& center, const Vector& normal,
			  const double size1, const double size2 )
{
  Vector axis1, axis2;
  normal.find_orthogonal(axis1, axis2);
   
  variables[PointRVar]->Move(center+axis1*size1);
  variables[PointDVar]->Move(center+axis2*size2);
  variables[CenterVar]->Move(center);
  variables[DistRVar]->Move(size1);
  variables[DistDVar]->Set(size2); // This should set the Hypo...
  variables[SDistRVar]->Set(size1*variables[RatioRVar]->real(), Scheme1); // Slider1...
  variables[SDistDVar]->Set(size2*variables[RatioDVar]->real(), Scheme1); // Slider2...

  execute(0);
}


void
FrameWidget::GetPosition( Point& center, Vector& normal,
			  double& size1, double& size2 )
{
  center = variables[CenterVar]->point();
  normal = Cross(GetRightAxis(), GetDownAxis());
  size1 = variables[DistRVar]->real();
  size2 = variables[DistDVar]->real();
}


void
FrameWidget::SetRatioR( const double ratio )
{
  ASSERT((ratio>=0.0) && (ratio<=1.0));
  variables[RatioRVar]->Set(ratio);
   
  execute(0);
}


double
FrameWidget::GetRatioR() const
{
  return (variables[RatioRVar]->real());
}


void
FrameWidget::SetRatioD( const double ratio )
{
  ASSERT((ratio>=0.0) && (ratio<=1.0));
  variables[RatioDVar]->Set(ratio);
   
  execute(0);
}


double
FrameWidget::GetRatioD() const
{
  return (variables[RatioDVar]->real());
}


void
FrameWidget::SetSize( const double sizeR, const double sizeD )
{
  ASSERT((sizeR>=0.0)&&(sizeD>=0.0));

  Point center(variables[CenterVar]->point());
  Vector axisR(variables[PointRVar]->point() - center);
  Vector axisD(variables[PointDVar]->point() - center);
  double ratioR(sizeR/variables[DistRVar]->real());
  double ratioD(sizeD/variables[DistDVar]->real());

  variables[PointRVar]->Move(center+axisR*ratioR);
  variables[PointDVar]->Move(center+axisD*ratioD);

  variables[DistRVar]->Move(sizeR);
  variables[DistDVar]->Set(sizeD); // This should set the Hypo...
  variables[SDistRVar]->Set(sizeR*variables[RatioRVar]->real(), Scheme1); // Slider1...
  variables[SDistDVar]->Set(sizeD*variables[RatioDVar]->real(), Scheme1); // Slider2...

  execute(0);
}

void
FrameWidget::GetSize( double& sizeR, double& sizeD ) const
{
  sizeR = variables[DistRVar]->real();
  sizeD = variables[DistDVar]->real();
}


const Vector&
FrameWidget::GetRightAxis()
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
FrameWidget::GetDownAxis()
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


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
FrameWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());
   
  switch(mindex)
  {
  case 0:
    return "Point";
  case 1:
    return "Edge";
  case 2:
    return "Resize";
  case 3:
    return "Slider";
  default:
    return "UnknownMaterial";
  }
}

} // End namespace SCIRun
