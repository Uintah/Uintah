
/*
 *  LightWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/LightWidget.h>
#include <Widgets/FrameWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/ProjectConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cone.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>
#include <Geom/Torus.h>
#include <Malloc/Allocator.h>

const Index NumCons = 4;
const Index NumVars = 6;
const Index NumGeoms = 7;
const Index NumPcks = 5;
const Index NumMatls = 4;
const Index NumMdes = 4;
const Index NumSwtchs = 4;
const Index NumSchemes = 4;

enum { ConstAdjacent, ConstOpposite, ConstRatio, ConstProject };
enum { GeomSource, GeomDirect, GeomCone, GeomAxis, GeomRing, GeomHead, GeomShaft };
enum { PickSource, PickDirect, PickCone, PickAxis, PickArrow };

LightWidget::LightWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, "LightWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  ltype(DirectionalLight), oldaxis(1, 0, 0)
{
   Real INIT = 10.0*widget_scale;
   // Scheme4 is used for the Arrow.
   variables[SourceVar] = scinew PointVariable("Source", solve, Scheme1, Point(0, 0, 0));
   variables[DirectVar] = scinew PointVariable("Direct", solve, Scheme2, Point(INIT, 0, 0));
   variables[ConeVar] = scinew PointVariable("Cone", solve, Scheme3, Point(INIT, INIT, 0));
   variables[DistVar] = scinew RealVariable("Dist", solve, Scheme1, INIT);
   variables[RadiusVar] = scinew RealVariable("Radius", solve, Scheme1, INIT);
   variables[RatioVar] = scinew RealVariable("Ratio", solve, Scheme1, 1.0);

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
   picks[PickArrow] = scinew GeomPick(materials[ArrowMatl], module, this, PickArrow);
   picks[PickArrow]->set_highlight(DefaultHighlightMaterial);
   CreateModeSwitch(0, picks[PickArrow]);

   geometries[GeomSource] = scinew GeomSphere;
   picks[PickSource] = scinew GeomPick(geometries[GeomSource], module, this, PickSource);
   picks[PickSource]->set_highlight(DefaultHighlightMaterial);
   materials[SourceMatl] = scinew GeomMaterial(picks[PickSource], DefaultPointMaterial);
   CreateModeSwitch(1, materials[SourceMatl]);

   GeomGroup* spheres = scinew GeomGroup;
   geometries[GeomDirect] = scinew GeomSphere;
   picks[PickDirect] = scinew GeomPick(geometries[GeomDirect], module, this, PickDirect);
   picks[PickDirect]->set_highlight(DefaultHighlightMaterial);
   spheres->add(picks[PickDirect]);

   geometries[GeomCone] = scinew GeomSphere;
   picks[PickCone] = scinew GeomPick(geometries[GeomCone], module, this, PickCone);
   picks[PickCone]->set_highlight(DefaultHighlightMaterial);
   spheres->add(picks[PickCone]);
   materials[PointMatl] = scinew GeomMaterial(spheres, DefaultPointMaterial);

   GeomGroup* axes = scinew GeomGroup;
   geometries[GeomAxis] = scinew GeomCylinder;
   axes->add(geometries[GeomAxis]);
   geometries[GeomRing] = scinew GeomTorus;
   axes->add(geometries[GeomRing]);
   picks[PickAxis] = scinew GeomPick(axes, module, this, PickAxis);
   picks[PickAxis]->set_highlight(DefaultHighlightMaterial);
   materials[ConeMatl] = scinew GeomMaterial(picks[PickAxis], DefaultEdgeMaterial);

   GeomGroup* conegroup = scinew GeomGroup;
   conegroup->add(materials[PointMatl]);
   conegroup->add(materials[ConeMatl]);
   CreateModeSwitch(2, conegroup);

   arealight = scinew FrameWidget(module, lock, widget_scale);
   CreateModeSwitch(3, arealight->GetWidget());

   SetMode(Mode0, Switch1|Switch0);
   SetMode(Mode1, Switch1);
   SetMode(Mode2, Switch1|Switch2);
   SetMode(Mode3, Switch3);

   FinishWidget();
}


LightWidget::~LightWidget()
{
}


void
LightWidget::redraw()
{
   Real sphererad(widget_scale), cylinderrad(0.5*widget_scale);
   Point center(variables[SourceVar]->point());
   Vector direct(GetAxis());
   
   if (mode_switches[0]->get_state()) {
      ((GeomCylinder*)geometries[GeomShaft])->move(center,
						   center + direct * 3.0 * widget_scale,
						   0.5 * widget_scale);
      ((GeomCappedCone*)geometries[GeomHead])->move(center + direct * 3.0 * widget_scale,
						    center + direct * 5.0 * widget_scale,
						    widget_scale,
						    0);
   }
   
   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[GeomSource])->move(center, sphererad);
   }

   if (mode_switches[2]->get_state()) {
      ((GeomSphere*)geometries[GeomDirect])->move(variables[DirectVar]->point(), sphererad);
      ((GeomSphere*)geometries[GeomCone])->move(variables[ConeVar]->point(), sphererad);
      ((GeomCylinder*)geometries[GeomAxis])->move(variables[SourceVar]->point(), variables[DirectVar]->point(),
						  cylinderrad);
      ((GeomTorus*)geometries[GeomRing])->move(variables[DirectVar]->point(), direct,
					       variables[RadiusVar]->real(), cylinderrad);
   }

   if (mode_switches[3]->get_state()) {
      arealight->SetScale(widget_scale);
      arealight->execute(0);
   }

   if (direct.length2() > 1e-6) {
      direct.normalize();
      Vector v1, v2;
      direct.find_orthogonal(v1, v2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == PickCone)
	    picks[geom]->set_principal(v1, v2);
	 else
	    picks[geom]->set_principal(direct, v1, v2);
      }
   }
}

void
LightWidget::geom_moved( GeomPick* gp, int axis, double dist,
			 const Vector& delta, int pick, const BState& state )
{
   switch(ltype) {
   case DirectionalLight:
   case PointLight:
   case SpotLight:
      switch(pick) {
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
   case AreaLight:
      arealight->geom_moved(gp, axis, dist, delta, pick, state);
      break;
   }
   
   execute(0);
}


void
LightWidget::MoveDelta( const Vector& delta )
{
   arealight->MoveDelta(delta);

   variables[SourceVar]->MoveDelta(delta);
   variables[DirectVar]->MoveDelta(delta);
   variables[ConeVar]->MoveDelta(delta);

   execute(1);
}


Point
LightWidget::ReferencePoint() const
{
   return variables[SourceVar]->point();
}


void
LightWidget::SetLightType( const LightType lighttype )
{
   ltype = lighttype;

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
      return oldaxis;
   else 
      return (oldaxis = axis.normal());
}


void
LightWidget::NextMode()
{
   Real s1, s2;
   switch(ltype) {
   case DirectionalLight:
   case PointLight:
   case SpotLight:
      arealight->GetSize(s1, s2);
      arealight->SetPosition(variables[SourceVar]->point(), GetAxis(), s1, s2);
      break;
   case AreaLight:
      Point center;
      Vector normal;
      arealight->GetPosition(center, normal, s1, s2);

      variables[SourceVar]->Move(center);
      variables[DirectVar]->Set(center+normal, Scheme4);
      break;
   }
   
   Index s;
   for (s=0; s<NumSwitches; s++)
      if (modes[CurrentMode]&(1<<s))
	 mode_switches[s]->set_state(0);
   CurrentMode = (CurrentMode+1) % NumModes;
   ltype = (LightType)((ltype+1) % NumLightTypes);
   for (s=0; s<NumSwitches; s++)
      if (modes[CurrentMode]&(1<<s))
	 mode_switches[s]->set_state(1);

   execute(0);
}


clString
LightWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   
   switch(mindex){
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


