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
 *  LightWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Dataflow/Widgets/LightWidget.h>
#include <Dataflow/Widgets/FrameWidget.h>
#include <Dataflow/Constraints/DistanceConstraint.h>
#include <Dataflow/Constraints/ProjectConstraint.h>
#include <Dataflow/Constraints/RatioConstraint.h>
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomTorus.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {



const Index NumCons = 4;
const Index NumVars = 6;
const Index NumGeoms = 7;
const Index NumPcks = 5;
const Index NumMatls = 4;
const Index NumMdes = 3;
const Index NumSwtchs = 3;
const Index NumSchemes = 4;

enum { ConstAdjacent, ConstOpposite, ConstRatio, ConstProject };
enum { GeomSource, GeomDirect, GeomCone, GeomAxis, GeomRing, GeomHead, GeomShaft };
enum { PickSource, PickDirect, PickCone, PickAxis, PickArrow };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
LightWidget::LightWidget( Module* module, CrowdMonitor* lock, 
			  double widget_scale )
  : BaseWidget(module, lock, "LightWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
    ltype(DirectionalLight), oldaxis(1, 0, 0)
{
  const double INIT = 10.0*widget_scale_;
  // Scheme4 is used for the Arrow.
  variables[SourceVar] = scinew PointVariable("Source", solve, Scheme1, Point(0, 0, 0));
  variables[DirectVar] = scinew PointVariable("Direct", solve, Scheme2, Point(INIT, 0, 0));
  variables[ConeVar] = scinew PointVariable("Cone", solve, Scheme3, Point(INIT, INIT, 0));
  variables[DistVar] = scinew RealVariable("Dist", solve, Scheme1, INIT);
  variables[RadiusVar] = scinew RealVariable("Radius", solve, Scheme1, INIT);
  variables[RatioVar] = scinew RealVariable("Ratio", solve, Scheme1, 1.0);
  init( module );
}

LightWidget::LightWidget( Module* module, CrowdMonitor* lock, 
			  double widget_scale, Point source, 
			  Point direct, Point cone, double rad, double rat)
  : BaseWidget(module, lock, "LightWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
    ltype(DirectionalLight), oldaxis(1, 0, 0)
{
  const double INIT = 10.0*widget_scale_;
  // Scheme4 is used for the Arrow.
  variables[SourceVar] = scinew PointVariable("Source", solve, Scheme1, source);
  variables[DirectVar] = scinew PointVariable("Direct", solve, Scheme2, direct);
  variables[ConeVar] = scinew PointVariable("Cone", solve, Scheme3, cone );
  variables[DistVar] = scinew RealVariable("Dist", solve, Scheme1, INIT);
  variables[RadiusVar] = scinew RealVariable("Radius", solve, Scheme1, rad);
  variables[RatioVar] = scinew RealVariable("Ratio", solve, Scheme1, rat);
  init( module );
}

void
LightWidget::init( Module* module)
{
  constraints[ConstAdjacent] = scinew DistanceConstraint("ConstAdjacent",
							 NumSchemes,
							 variables[SourceVar],
							 variables[DirectVar],
							 variables[DistVar]);
  constraints[ConstAdjacent]->VarChoices(Scheme1, 1, 1, 1);
  constraints[ConstAdjacent]->VarChoices(Scheme2, 2, 2, 2);
  constraints[ConstAdjacent]->VarChoices(Scheme3, 1, 1, 1);
  constraints[ConstAdjacent]->VarChoices(Scheme4, 1, 1, 1);
  constraints[ConstAdjacent]->Priorities(P_Default, P_Default, P_Default);
  constraints[ConstOpposite] = scinew DistanceConstraint("ConstOpposite",
							 NumSchemes,
							 variables[ConeVar],
							 variables[DirectVar],
							 variables[RadiusVar]);
  constraints[ConstOpposite]->VarChoices(Scheme1, 0, 0, 0);
  constraints[ConstOpposite]->VarChoices(Scheme2, 0, 0, 0);
  constraints[ConstOpposite]->VarChoices(Scheme3, 2, 2, 2);
  constraints[ConstOpposite]->VarChoices(Scheme4, 0, 0, 0);
  constraints[ConstOpposite]->Priorities(P_Default, P_Default, P_Default);
  constraints[ConstRatio] = scinew RatioConstraint("ConstRatio",
						   NumSchemes,
						   variables[DistVar],
						   variables[RadiusVar],
						   variables[RatioVar]);
  constraints[ConstRatio]->VarChoices(Scheme1, 1, 1, 1);
  constraints[ConstRatio]->VarChoices(Scheme2, 1, 1, 1);
  constraints[ConstRatio]->VarChoices(Scheme3, 2, 2, 2);
  constraints[ConstRatio]->VarChoices(Scheme4, 1, 1, 1);
  constraints[ConstRatio]->Priorities(P_Highest, P_Highest, P_Highest);
  constraints[ConstProject] = scinew ProjectConstraint("ConstProject",
						       NumSchemes,
						       variables[DirectVar],
						       variables[ConeVar],
						       variables[DirectVar],
						       variables[SourceVar]);
  constraints[ConstProject]->VarChoices(Scheme1, 1, 1, 1, 1);
  constraints[ConstProject]->VarChoices(Scheme2, 1, 1, 1, 1);
  constraints[ConstProject]->VarChoices(Scheme3, 1, 1, 1, 1);
  constraints[ConstProject]->VarChoices(Scheme4, 1, 1, 1, 1);
  constraints[ConstProject]->Priorities(P_Default, P_Default, P_Default, P_Default);

  GeomGroup* arr = scinew GeomGroup;
  geometries[GeomShaft] = scinew GeomCylinder;
  arr->add(geometries[GeomShaft]);
  geometries[GeomHead] = scinew GeomCappedCone;
  arr->add(geometries[GeomHead]);
  materials[ArrowMatl] = scinew GeomMaterial(arr, DefaultEdgeMaterial);
  picks_[PickArrow] = scinew GeomPick(materials[ArrowMatl], module, this, PickArrow);
  picks(PickArrow)->set_highlight(DefaultHighlightMaterial);
  CreateModeSwitch(0, picks_[PickArrow]);

  geometries[GeomSource] = scinew GeomSphere;
  picks_[PickSource] = scinew GeomPick(geometries[GeomSource], module, this, PickSource);
  picks(PickSource)->set_highlight(DefaultHighlightMaterial);
  materials[SourceMatl] = scinew GeomMaterial(picks_[PickSource], DefaultPointMaterial);
  CreateModeSwitch(1, materials[SourceMatl]);

  GeomGroup* spheres = scinew GeomGroup;
  geometries[GeomDirect] = scinew GeomSphere;
  picks_[PickDirect] = scinew GeomPick(geometries[GeomDirect], module, this, PickDirect);
  picks(PickDirect)->set_highlight(DefaultHighlightMaterial);
  spheres->add(picks_[PickDirect]);

  geometries[GeomCone] = scinew GeomSphere;
  picks_[PickCone] = scinew GeomPick(geometries[GeomCone], module, this, PickCone);
  picks(PickCone)->set_highlight(DefaultHighlightMaterial);
  spheres->add(picks_[PickCone]);
  materials[PointMatl] = scinew GeomMaterial(spheres, DefaultPointMaterial);

  GeomGroup* axes = scinew GeomGroup;
  geometries[GeomAxis] = scinew GeomCylinder;
  axes->add(geometries[GeomAxis]);
  geometries[GeomRing] = scinew GeomTorus;
  axes->add(geometries[GeomRing]);
  picks_[PickAxis] = scinew GeomPick(axes, module, this, PickAxis);
  picks(PickAxis)->set_highlight(DefaultHighlightMaterial);
  materials[ConeMatl] = scinew GeomMaterial(picks_[PickAxis], DefaultEdgeMaterial);

  GeomGroup* conegroup = scinew GeomGroup;
  conegroup->add(materials[PointMatl]);
  conegroup->add(materials[ConeMatl]);
  CreateModeSwitch(2, conegroup);

//   arealight = scinew FrameWidget(module, lock, widget_scale);
//   CreateModeSwitch(3, arealight->GetWidget());

  SetMode(Mode0, Switch1|Switch0);
  SetMode(Mode1, Switch1);
  SetMode(Mode2, Switch1|Switch2);
//   SetMode(Mode3, Switch3);

  FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
LightWidget::~LightWidget()
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
LightWidget::redraw()
{
  const double sphererad(widget_scale_);
  const double cylinderrad(0.5*widget_scale_);
  Point center(variables[SourceVar]->point());
  Vector direct(GetAxis());
   
  if (mode_switches[0]->get_state())
  {
    geometry<GeomCylinder*>(GeomShaft)->move(center,
					     center + direct * 3.0 * widget_scale_,
					     0.5 * widget_scale_);
    geometry<GeomCappedCone*>(GeomHead)->move(center + direct * 3.0 * widget_scale_,
					      center + direct * 5.0 * widget_scale_,
					      widget_scale_,
					      0);
  }
   
  if (mode_switches[1]->get_state())
  {
    geometry<GeomSphere*>(GeomSource)->move(center, sphererad);
  }

  if (mode_switches[2]->get_state())
  {
    geometry<GeomSphere*>(GeomDirect)->move(variables[DirectVar]->point(), sphererad);
    geometry<GeomSphere*>(GeomCone)->move(variables[ConeVar]->point(), sphererad);
    geometry<GeomCylinder*>(GeomAxis)->move(variables[SourceVar]->point(), variables[DirectVar]->point(),
					    cylinderrad);
    geometry<GeomTorus*>(GeomRing)->move(variables[DirectVar]->point(), direct,
					 variables[RadiusVar]->real(), cylinderrad);
  }

//    if (mode_switches[3]->get_state())
//    {
//      arealight->SetScale(widget_scale_);
//    }

  if (direct.length2() > 1e-6)
  {
    direct.normalize();
    Vector v1, v2;
    direct.find_orthogonal(v1, v2);
    for (Index geom = 0; geom < NumPcks; geom++)
    {
      if (geom == PickCone)
      {
	picks(geom)->set_principal(v1, v2);
      }
      else
      {
	picks(geom)->set_principal(direct, v1, v2);
      }
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
LightWidget::geom_moved( GeomPickHandle gp, int axis, double dist,
			 const Vector& delta, int pick, const BState& state,
			 const Vector &pick_offset)
{
  switch(ltype)
  {
  case DirectionalLight:
  case PointLight:
  case SpotLight:
    switch(pick)
    {
    case PickSource:
      variables[SourceVar]->SetDelta(delta);
      break;

    case PickArrow:
      variables[DirectVar]->SetDelta(delta, Scheme4);
      break;

    case PickDirect:
      variables[DirectVar]->SetDelta(delta);
      break;

    case PickCone:
      variables[ConeVar]->SetDelta(delta);
      break;

    case PickAxis:
      variables[SourceVar]->MoveDelta(delta);
      variables[DirectVar]->MoveDelta(delta);
      variables[ConeVar]->MoveDelta(delta);
      break;
    }
    break;

//   case AreaLight:
//     arealight->geom_moved(gp, axis, dist, delta, pick, state, pick_offset);
//     break;

  default:
    ASSERTFAIL("Bad light type");
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
LightWidget::MoveDelta( const Vector& delta )
{
//   arealight->MoveDelta(delta);

  variables[SourceVar]->MoveDelta(delta);
  variables[DirectVar]->MoveDelta(delta);
  variables[ConeVar]->MoveDelta(delta);

  execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
LightWidget::ReferencePoint() const
{
  return variables[SourceVar]->point();
}


void
LightWidget::SetLightType( const LightType lighttype )
{
  Index s;
  for (s=0; s<mode_switches.size(); s++)
  {
    if (modes[current_mode_]&(1<<s))
    {
      mode_switches[s]->set_state(0);
    }
  }
  current_mode_ = ((Index) lighttype) % modes.size();
  ltype = lighttype;
  for (s=0; s<mode_switches.size(); s++)
  {
    if (modes[current_mode_]&(1<<s))
    {
      mode_switches[s]->set_state(1);
    }
  }

  execute(0);

}


LightType
LightWidget::GetLightType() const
{
  return ltype;
}


const Vector&
LightWidget::GetAxis()
{
  Vector axis(variables[DirectVar]->point() - variables[SourceVar]->point());
  if (axis.length2() <= 1e-6)
  {
    return oldaxis;
  }
  else
  {
    return (oldaxis = axis.normal());
  }
}
Point 
LightWidget::GetSource() const
{
  Point source(variables[SourceVar]->point());
  return source;
} 

Point
LightWidget::GetPointAt() const
{
  Point at(variables[DirectVar]->point());
  return at;
}
Point
LightWidget::GetCone() const
{
  Point at(variables[ConeVar]->point());
  return at;
}

void 
LightWidget::SetPointAt( const Point& pt)
{
  variables[DirectVar]->Set( pt, Scheme2 );
}
void 
LightWidget::SetCone( const Point& pt)
{
  variables[ConeVar]->Set( pt, Scheme3 );
}

double 
LightWidget::GetRadius() const
{
  return variables[RadiusVar]->real();
}

void
LightWidget::SetRadius( double rad )
{
  variables[RadiusVar]->Set( rad, Scheme1 );
}

double 
LightWidget::GetRatio() const
{
  return variables[RatioVar]->real();
}

void
LightWidget::SetRatio( double rat )
{
  variables[RatioVar]->Set( rat, Scheme1 );
}

void
LightWidget::NextMode()
{
  //  double s1, s2;
  switch(ltype)
  {
  case DirectionalLight:
  case PointLight:
  case SpotLight:
//     arealight->GetSize(s1, s2);
//     arealight->SetPosition(variables[SourceVar]->point(), GetAxis(), s1, s2);
    break;

//   case AreaLight:
//     {
//       Point center;
//       Vector normal;
//       arealight->GetPosition(center, normal, s1, s2);

//       variables[SourceVar]->Move(center);
//       variables[DirectVar]->Set(center+normal, Scheme4);
//       break;
//     }

  default:
    ASSERTFAIL("Bad Light Type");
  }
   
  Index s;
  for (s=0; s<mode_switches.size(); s++)
  {
    if (modes[current_mode_]&(1<<s))
    {
      mode_switches[s]->set_state(0);
    }
  }
  current_mode_ = (current_mode_+1) % modes.size();
  ltype = (LightType)((ltype+1) % AreaLight);
  for (s=0; s<mode_switches.size(); s++)
  {
    if (modes[current_mode_]&(1<<s))
    {
      mode_switches[s]->set_state(1);
    }
  }

  execute(0);
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
LightWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());
   
  switch(mindex)
  {
  case 0:
    return "Source";
  case 1:
    return "Arrow";
  case 2:
    return "Point";
  case 3:
    return "Cone";
  default:
    return "UnknownMaterial";
  }
}


} // End namespace SCIRun

