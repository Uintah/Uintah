
/*
 *  RingWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/RingWidget.h>
#include <Constraints/AngleConstraint.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>
#include <Geom/Torus.h>
#include <Geometry/Plane.h>
#include <Math/Expon.h>
#include <Math/Trig.h>

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

RingWidget::RingWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, "RingWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  oldrightaxis(1, 0, 0), olddownaxis(0, 1, 0)
{
   Real INIT = 5.0*widget_scale;
   // Scheme4/5 are used to resize.
   variables[PointRVar] = new PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
   variables[PointDVar] = new PointVariable("PntD", solve, Scheme2, Point(0, INIT, 0));
   variables[CenterVar] = new PointVariable("Center", solve, Scheme1, Point(0, 0, 0));
   variables[SliderVar] = new PointVariable("Slider", solve, Scheme3, Point(INIT, 0, 0));
   variables[DistVar] = new RealVariable("Dist", solve, Scheme1, INIT);
   variables[HypoVar] = new RealVariable("HYPO", solve, Scheme1, sqrt(2*INIT*INIT));
   variables[Sqrt2Var] = new RealVariable("Sqrt2", solve, Scheme1, sqrt(2));
   variables[AngleVar] = new RealVariable("Angle", solve, Scheme3, 0);

   constraints[ConstAngle] = new AngleConstraint("ConstAngle",
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
   constraints[ConstRadius] = new DistanceConstraint("ConstRadius",
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
   constraints[ConstRD] = new DistanceConstraint("ConstRD",
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
   constraints[ConstPyth] = new RatioConstraint("ConstPyth",
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
   constraints[ConstRC] = new DistanceConstraint("ConstRC",
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
   constraints[ConstDC] = new DistanceConstraint("ConstDC",
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
   geometries[GeomRing] = new GeomTorus;
   picks[PickRing] = new GeomPick(geometries[GeomRing], module, this, PickRing);
   picks[PickRing]->set_highlight(DefaultHighlightMaterial);
   materials[RingMatl] = new GeomMaterial(picks[PickRing], DefaultEdgeMaterial);
   CreateModeSwitch(0, materials[RingMatl]);

   GeomGroup* pts = new GeomGroup;
   for (geom = GeomPointU, pick = PickSphU;
	geom <= GeomPointL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(DefaultHighlightMaterial);
      pts->add(picks[pick]);
   }
   materials[PointMatl] = new GeomMaterial(pts, DefaultPointMaterial);
   
   geometries[GeomResizeL] = new GeomCappedCylinder;
   picks[PickResizeL] = new GeomPick(geometries[GeomResizeL],
				     module, this, PickResizeL);
   picks[PickResizeL]->set_highlight(DefaultHighlightMaterial);
   materials[HalfResizeMatl] = new GeomMaterial(picks[PickResizeL], DefaultSpecialMaterial);

   GeomGroup* resulr = new GeomGroup;
   geometries[GeomResizeU] = new GeomCappedCylinder;
   picks[PickResizeU] = new GeomPick(geometries[GeomResizeU],
				     module, this, PickResizeU);
   picks[PickResizeU]->set_highlight(DefaultHighlightMaterial);
   resulr->add(picks[PickResizeU]);
   geometries[GeomResizeD] = new GeomCappedCylinder;
   picks[PickResizeD] = new GeomPick(geometries[GeomResizeD],
				     module, this, PickResizeD);
   picks[PickResizeD]->set_highlight(DefaultHighlightMaterial);
   resulr->add(picks[PickResizeD]);
   geometries[GeomResizeR] = new GeomCappedCylinder;
   picks[PickResizeR] = new GeomPick(geometries[GeomResizeR],
				     module, this, PickResizeR);
   picks[PickResizeR]->set_highlight(DefaultHighlightMaterial);
   resulr->add(picks[PickResizeR]);
   materials[ResizeMatl] = new GeomMaterial(resulr, DefaultResizeMaterial);

   GeomGroup* w = new GeomGroup;
   w->add(materials[PointMatl]);
   w->add(materials[HalfResizeMatl]);
   w->add(materials[ResizeMatl]);
   CreateModeSwitch(1, w);

   geometries[GeomSlider] = new GeomCappedCylinder;
   picks[PickSlider] = new GeomPick(geometries[GeomSlider], module, this,
				    PickSlider);
   picks[PickSlider]->set_highlight(DefaultHighlightMaterial);
   materials[SliderMatl] = new GeomMaterial(picks[PickSlider], DefaultSliderMaterial);
   CreateModeSwitch(2, materials[SliderMatl]);

   SetMode(Mode0, Switch0|Switch1|Switch2);
   SetMode(Mode1, Switch0|Switch1);
   SetMode(Mode2, Switch0|Switch2);
   SetMode(Mode3, Switch0);

   FinishWidget();
}


RingWidget::~RingWidget()
{
}


void
RingWidget::widget_execute()
{
   Vector Right(GetRightAxis()*variables[DistVar]->real());
   Vector Down(GetDownAxis()*variables[DistVar]->real());
   Point Center(variables[CenterVar]->point());
   Point U(Center-Down);
   Point R(Center+Right);
   Point D(Center+Down);
   Point L(Center-Right);
   
   Vector normal(Cross(GetRightAxis(), GetDownAxis()));
   if (normal.length2() < 1e-6) normal = Vector(1.0, 0.0, 0.0);
   
   if (mode_switches[0]->get_state()) {
      ((GeomTorus*)geometries[GeomRing])->move(variables[CenterVar]->point(), normal,
					       variables[DistVar]->real(), 0.5*widget_scale);
   }

   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[GeomPointU])->move(U, widget_scale);
      ((GeomSphere*)geometries[GeomPointR])->move(R, widget_scale);
      ((GeomSphere*)geometries[GeomPointD])->move(D, widget_scale);
      ((GeomSphere*)geometries[GeomPointL])->move(L, widget_scale);
      ((GeomCappedCylinder*)geometries[GeomResizeU])->move(U, U - (GetDownAxis() * 1.5 * widget_scale),
							   0.5*widget_scale);
      ((GeomCappedCylinder*)geometries[GeomResizeR])->move(R, R + (GetRightAxis() * 1.5 * widget_scale),
							   0.5*widget_scale);
      ((GeomCappedCylinder*)geometries[GeomResizeD])->move(D, D + (GetDownAxis() * 1.5 * widget_scale),
							   0.5*widget_scale);
      ((GeomCappedCylinder*)geometries[GeomResizeL])->move(L, L - (GetRightAxis() * 1.5 * widget_scale),
							   0.5*widget_scale);
   }

   Vector slide(variables[SliderVar]->point()-variables[CenterVar]->point());
   if (slide.length2() > 1e-6) {
      slide = Cross(normal, slide.normal());
      if (slide.length2() < 1e-6) slide = Vector(0.0, 1.0, 0.0);
   } else {
      slide = GetRightAxis();
   }
   if (mode_switches[2]->get_state()) {
      ((GeomCappedCylinder*)geometries[GeomSlider])->move(variables[SliderVar]->point()
							  - (slide * 0.3 * widget_scale),
							  variables[SliderVar]->point()
							  + (slide * 0.3 * widget_scale),
							  1.1*widget_scale);
   }
   
   ((DistanceConstraint*)constraints[ConstRC])->SetMinimum(1.6*widget_scale);
   ((DistanceConstraint*)constraints[ConstDC])->SetMinimum(1.6*widget_scale);
   ((DistanceConstraint*)constraints[ConstRD])->SetMinimum(sqrt(2*1.6*1.6)*widget_scale);

   Right.normalize();
   Down.normalize();
   Vector Norm(Cross(Right, Down));
   for (Index geom = 0; geom < NumPcks; geom++) {
      if (geom == PickSlider)
	 picks[geom]->set_principal(slide);
      else if ((geom == PickResizeU) || (geom == PickResizeD))
	 picks[geom]->set_principal(Down);
      else if ((geom == PickResizeL) || (geom == PickResizeR))
	 picks[geom]->set_principal(Right);
      else if ((geom == PickSphR) || (geom == PickSphL))
	 picks[geom]->set_principal(Down, Norm);
      else if ((geom == PickSphU) || (geom == PickSphD))
	 picks[geom]->set_principal(Right, Norm);
      else
	 picks[geom]->set_principal(Right, Down, Norm);
   }
}


void
RingWidget::geom_moved( int axis, double dist, const Vector& delta,
		        int pick, const BState& )
{
   Point p;
   Real ResizeMin(1.5*widget_scale);
   if (axis==1) dist = -dist;

   ((DistanceConstraint*)constraints[ConstRC])->SetDefault(GetRightAxis());
   ((DistanceConstraint*)constraints[ConstDC])->SetDefault(GetDownAxis());

   switch(pick){
   case PickSphU:
      variables[PointDVar]->SetDelta(-delta);
      break;
   case PickSphR:
      variables[PointRVar]->SetDelta(delta);
      break;
   case PickSphD:
      variables[PointDVar]->SetDelta(delta);
      break;
   case PickSphL:
      variables[PointRVar]->SetDelta(-delta);
      break;
   case PickResizeU:
      if ((variables[DistVar]->real() - dist) < ResizeMin)
	 p = variables[CenterVar]->point() + GetDownAxis()*ResizeMin;
      else
	 p = variables[PointDVar]->point() - delta;
      variables[PointDVar]->Set(p, Scheme5);
      break;
   case PickResizeR:
      if ((variables[DistVar]->real() + dist) < ResizeMin)
	 p = variables[CenterVar]->point() + GetRightAxis()*ResizeMin;
      else
	 p = variables[PointRVar]->point() + delta;
      variables[PointRVar]->Set(p, Scheme4);
      break;
   case PickResizeD:
      if ((variables[DistVar]->real() + dist) < ResizeMin)
	 p = variables[CenterVar]->point() + GetDownAxis()*ResizeMin;
      else
	 p = variables[PointDVar]->point() + delta;
      variables[PointDVar]->Set(p, Scheme5);
      break;
   case PickResizeL:
      if ((variables[DistVar]->real() - dist) < ResizeMin)
	 p = variables[CenterVar]->point() + GetRightAxis()*ResizeMin;
      else
	 p = variables[PointRVar]->point() - delta;
      variables[PointRVar]->Set(p, Scheme4);
      break;
   case PickSlider:
      variables[SliderVar]->SetDelta(delta);
      break;
   case PickRing:
      MoveDelta(delta);
      break;
   }
   execute();
}


void
RingWidget::MoveDelta( const Vector& delta )
{
   variables[PointRVar]->MoveDelta(delta);
   variables[PointDVar]->MoveDelta(delta);
   variables[CenterVar]->MoveDelta(delta);
   variables[SliderVar]->MoveDelta(delta);

   execute();
}


Point
RingWidget::ReferencePoint() const
{
   return variables[CenterVar]->point();
}


void
RingWidget::SetPosition( const Point& center, const Vector& normal, const Real radius )
{
   Vector v1, v2;
   normal.find_orthogonal(v1, v2);
   variables[CenterVar]->Move(center);
   variables[PointRVar]->Move(center+v1*radius);
   variables[PointDVar]->Set(center+v2*radius);

   execute();
}


void
RingWidget::GetPosition( Point& center, Vector& normal, Real& radius ) const
{
   center = variables[CenterVar]->point();
   normal = Plane(variables[PointRVar]->point(),
		  variables[PointDVar]->point(),
		  variables[CenterVar]->point()).normal();
   radius = variables[DistVar]->real();
}


void
RingWidget::SetRatio( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[AngleVar]->Set(ratio*2.0*Pi - Pi);

   execute();
}


Real
RingWidget::GetRatio() const
{
   Real ratio=variables[AngleVar]->real() / (2.0 * Pi);
   if(ratio < 0)
       ratio+=Pi;
   return ratio;
}


void
RingWidget::SetRadius( const Real radius )
{
   ASSERT(radius>=0.0);
   
   Vector axis1(variables[PointRVar]->point() - variables[CenterVar]->point());
   Vector axis2(variables[PointDVar]->point() - variables[CenterVar]->point());
   Real ratio(radius/variables[DistVar]->real());

   variables[PointRVar]->Move(variables[CenterVar]->point()+axis1*ratio);
   variables[PointDVar]->Move(variables[CenterVar]->point()+axis2*ratio);

   variables[DistVar]->Set(variables[DistVar]->real()*ratio); // This should set the slider...

   execute();
}


Real
RingWidget::GetRadius() const
{
   return variables[DistVar]->real();
}


const Vector&
RingWidget::GetRightAxis()
{
   Vector axis(variables[PointRVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldrightaxis;
   else
      return (oldrightaxis = axis.normal());
}


const Vector&
RingWidget::GetDownAxis()
{
   Vector axis(variables[PointDVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return olddownaxis;
   else
      return (olddownaxis = axis.normal());
}


void
RingWidget::GetPlane(Vector& v1, Vector& v2)
{
    v1=GetRightAxis();
    v2=GetDownAxis();
}


clString
RingWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   
   switch(mindex){
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


