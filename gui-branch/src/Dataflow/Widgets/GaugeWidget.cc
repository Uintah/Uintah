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
 *  GaugeWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Dataflow/Widgets/GaugeWidget.h>
#include <Dataflow/Constraints/DistanceConstraint.h>
#include <Dataflow/Constraints/RatioConstraint.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {



const Index NumCons = 2;
const Index NumVars = 5;
const Index NumGeoms = 6;
const Index NumPcks = 6;
const Index NumMatls = 4;
const Index NumMdes = 2;
const Index NumSwtchs = 2;
const Index NumSchemes = 3;

enum { ConstDist, ConstRatio };
enum { GeomPointL, GeomPointR, GeomShaft, GeomSlider, GeomResizeL, GeomResizeR };
enum { PickSphL, PickSphR, PickCyl, PickSlider, PickResizeL, PickResizeR };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
GaugeWidget::GaugeWidget( Module* module, CrowdMonitor* lock, double widget_scale )
  : BaseWidget(module, lock, "GaugeWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
    oldaxis(1, 0, 0)
{
  double INIT = 10.0*widget_scale;
  // Scheme3 is for resizing.
  variables[PointLVar] = scinew PointVariable("PntL", solve, Scheme1, Point(0, 0, 0));
  variables[PointRVar] = scinew PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
  variables[DistVar] = scinew RealVariable("Dist", solve, Scheme1, INIT);
  variables[SDistVar] = scinew RealVariable("SDistVar", solve, Scheme2, INIT/2.0);
  variables[RatioVar] = scinew RealVariable("Ratio", solve, Scheme1, 0.5);
   
  constraints[ConstDist] = scinew DistanceConstraint("ConstDist",
						     NumSchemes,
						     variables[PointLVar],
						     variables[PointRVar],
						     variables[DistVar]);
  constraints[ConstDist]->VarChoices(Scheme1, 1, 0, 1);
  constraints[ConstDist]->VarChoices(Scheme2, 1, 0, 1);
  constraints[ConstDist]->VarChoices(Scheme3, 2, 2, 2);
  constraints[ConstDist]->Priorities(P_Highest, P_Highest, P_Default);
  constraints[ConstRatio] = scinew RatioConstraint("ConstRatio",
						   NumSchemes,
						   variables[SDistVar],
						   variables[DistVar],
						   variables[RatioVar]);
  constraints[ConstRatio]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstRatio]->VarChoices(Scheme2, 2, 2, 2);
  constraints[ConstRatio]->VarChoices(Scheme3, 0, 0, 0);
  constraints[ConstRatio]->Priorities(P_Highest, P_Highest, P_Highest);

  geometries[GeomShaft] = scinew GeomCappedCylinder;
  picks[PickCyl] = scinew GeomPick(geometries[GeomShaft], module, this, PickCyl);
  picks[PickCyl]->set_highlight(DefaultHighlightMaterial);
  materials[ShaftMatl] = scinew GeomMaterial(picks[PickCyl], DefaultEdgeMaterial);
  CreateModeSwitch(0, materials[ShaftMatl]);

  GeomGroup* sphs = scinew GeomGroup;
  geometries[GeomPointL] = scinew GeomSphere;
  picks[PickSphL] = scinew GeomPick(geometries[GeomPointL],
				    module, this, PickSphL);
  picks[PickSphL]->set_highlight(DefaultHighlightMaterial);
  sphs->add(picks[PickSphL]);
  geometries[GeomPointR] = scinew GeomSphere;
  picks[PickSphR] = scinew GeomPick(geometries[GeomPointR],
				    module, this, PickSphR);
  picks[PickSphR]->set_highlight(DefaultHighlightMaterial);
  sphs->add(picks[PickSphR]);
  materials[PointMatl] = scinew GeomMaterial(sphs, DefaultPointMaterial);
   
  GeomGroup* resizes = scinew GeomGroup;
  geometries[GeomResizeL] = scinew GeomCappedCylinder;
  picks[PickResizeL] = scinew GeomPick(geometries[GeomResizeL],
				       module, this, PickResizeL);
  picks[PickResizeL]->set_highlight(DefaultHighlightMaterial);
  resizes->add(picks[PickResizeL]);
  geometries[GeomResizeR] = scinew GeomCappedCylinder;
  picks[PickResizeR] = scinew GeomPick(geometries[GeomResizeR],
				       module, this, PickResizeR);
  picks[PickResizeR]->set_highlight(DefaultHighlightMaterial);
  resizes->add(picks[PickResizeR]);
  materials[ResizeMatl] = scinew GeomMaterial(resizes, DefaultResizeMaterial);
   
  geometries[GeomSlider] = scinew GeomCappedCylinder;
  picks[PickSlider] = scinew GeomPick(geometries[GeomSlider],
				      module, this, PickSlider);
  picks[PickSlider]->set_highlight(DefaultHighlightMaterial);
  materials[SliderMatl] = scinew GeomMaterial(picks[PickSlider], DefaultSliderMaterial);
  GeomGroup* w = scinew GeomGroup;
  w->add(materials[PointMatl]);
  w->add(materials[ResizeMatl]);
  w->add(materials[SliderMatl]);
  CreateModeSwitch(1, w);

  SetMode(Mode0, Switch0|Switch1);
  SetMode(Mode1, Switch0);

  FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
GaugeWidget::~GaugeWidget()
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
GaugeWidget::redraw()
{
  Point L(variables[PointLVar]->point()), R(variables[PointRVar]->point()),
    S(L+GetAxis()*variables[SDistVar]->real());

  if (mode_switches[0]->get_state())
  {
    ((GeomCappedCylinder*)geometries[GeomShaft])->
      move(L, R, 0.5*widget_scale_);
  }

  if (mode_switches[1]->get_state())
  {
    ((GeomSphere*)geometries[GeomPointL])->move(L, widget_scale_);
    ((GeomSphere*)geometries[GeomPointR])->move(R, widget_scale_);
    ((GeomCappedCylinder*)geometries[GeomResizeL])->move(L, L - (GetAxis() * 1.5 * widget_scale_),
							 0.5*widget_scale_);
    ((GeomCappedCylinder*)geometries[GeomResizeR])->move(R, R + (GetAxis() * 1.5 * widget_scale_),
							 0.5*widget_scale_);
    ((GeomCappedCylinder*)geometries[GeomSlider])->move(S - (GetAxis() * 0.3 * widget_scale_),
							S + (GetAxis() * 0.3 * widget_scale_),
							1.1*widget_scale_);
  }
   
  Vector v(GetAxis()), v1, v2;
  v.find_orthogonal(v1,v2);
  for (Index geom = 0; geom < NumPcks; geom++)
  {
    if ((geom == PickSlider) || (geom == PickResizeL) || (geom == PickResizeR))
    {
      picks[geom]->set_principal(v);
    }
    else
    {
      picks[geom]->set_principal(v, v1, v2);
    }
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
GaugeWidget::geom_moved( GeomPick*, int axis, double dist,
			 const Vector& delta, int pick, const BState& )
{
  ((DistanceConstraint*)constraints[ConstDist])->SetDefault(GetAxis());

  switch(pick)
  {
  case PickSphL:
    variables[PointLVar]->SetDelta(delta);
    break;

  case PickSphR:
    variables[PointRVar]->SetDelta(delta);
    break;

  case PickResizeL:
    variables[PointLVar]->SetDelta(delta, Scheme3);
    break;

  case PickResizeR:
    variables[PointRVar]->SetDelta(delta, Scheme3);
    break;

  case PickSlider:
    {
      if (axis==1) { dist*=-1.0; }
      double sdist(variables[SDistVar]->real()+dist);
      if (sdist<0.0) { sdist=0.0; }
      else if (sdist>variables[DistVar]->real())
      {
	sdist=variables[DistVar]->real();
      }
      variables[SDistVar]->Set(sdist);
    }
    break;

  case PickCyl:
    MoveDelta(delta);
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
GaugeWidget::MoveDelta( const Vector& delta )
{
  variables[PointLVar]->MoveDelta(delta);
  variables[PointRVar]->MoveDelta(delta);

  execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
GaugeWidget::ReferencePoint() const
{
  return (variables[PointLVar]->point() +
	  (variables[PointRVar]->point() - variables[PointLVar]->point())*0.5);
}


void
GaugeWidget::SetRatio( const double ratio )
{
  ASSERT((ratio>=0.0) && (ratio<=1.0));
  variables[RatioVar]->Set(ratio);
   
  execute(0);
}


double
GaugeWidget::GetRatio() const
{
  return (variables[RatioVar]->real());
}


void
GaugeWidget::SetEndpoints( const Point& end1, const Point& end2 )
{
  variables[PointLVar]->Move(end1);
  variables[PointRVar]->Set(end2, Scheme3);
   
  execute(0);
}


void
GaugeWidget::GetEndpoints( Point& end1, Point& end2 ) const
{
  end1 = variables[PointLVar]->point();
  end2 = variables[PointRVar]->point();
}


const Vector&
GaugeWidget::GetAxis()
{
  Vector axis(variables[PointRVar]->point() - variables[PointLVar]->point());
  if (axis.length2() <= 1e-6)
    return oldaxis;
  else 
    return (oldaxis = axis.normal());
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
GaugeWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());
   
  switch(mindex)
  {
  case 0:
    return "Point";
  case 1:
    return "Shaft";
  case 2:
    return "Resize";
  case 3:
    return "Slider";
  default:
    return "UnknownMaterial";
  }
}

} // End namespace SCIRun


