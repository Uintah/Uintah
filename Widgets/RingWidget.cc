
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
#include <Constraints/MidpointConstraint.h>
#include <Constraints/PlaneConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>
#include <Geom/Torus.h>
#include <Geometry/Plane.h>

const Index NumCons = 13;
const Index NumVars = 11;
const Index NumGeoms = 7;
const Index NumMatls = 5;
const Index NumPcks = 7;
const Index NumSchemes = 4;

enum { ConstULDR, ConstURDL, ConstHypo, ConstPlane,
       ConstULUR, ConstULDL, ConstDRUR, ConstDRDL,
       ConstSDistVar, ConstCDist, ConstMidpoint, ConstSPlane,
       ConstAngle };
enum { GeomPointUL, GeomPointUR, GeomPointDR, GeomPointDL,
       GeomSlider, GeomRing, GeomResize };
enum { PickSphUL, PickSphUR, PickSphDR, PickSphDL, PickCyls,
       PickSlider, PickResize };

RingWidget::RingWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1),
  oldaxis1(1, 0, 0), oldaxis2(0, 1, 0)
{
   Real INIT = 1.0*widget_scale;
   // Scheme4 is used to resize.
   variables[PointULVar] = new PointVariable("PntUL", solve, Scheme1, Point(0, 0, 0));
   variables[PointURVar] = new PointVariable("PntUR", solve, Scheme2, Point(INIT, 0, 0));
   variables[PointDRVar] = new PointVariable("PntDR", solve, Scheme1, Point(INIT, INIT, 0));
   variables[PointDLVar] = new PointVariable("PntDL", solve, Scheme2, Point(0, INIT, 0));
   variables[CenterVar] = new PointVariable("Center", solve, Scheme2, Point(INIT/2.0, INIT/2.0, 0));
   variables[SliderVar] = new PointVariable("Slider", solve, Scheme3, Point(0, 0, 0));
   variables[DistVar] = new RealVariable("Dist", solve, Scheme1, INIT);
   variables[HypoVar] = new RealVariable("Hypo", solve, Scheme1, sqrt(2*INIT*INIT));
   variables[ConstVar] = new RealVariable("sqrt(2)", solve, Scheme1, sqrt(2));
   variables[SDistVar] = new RealVariable("SDistVar", solve, Scheme3, sqrt(INIT*INIT/2.0));
   variables[AngleVar] = new RealVariable("Angle", solve, Scheme1, 0);

   constraints[ConstAngle] = new AngleConstraint("ConstAngle",
						 NumSchemes,
						 variables[CenterVar],
						 variables[PointULVar],
						 variables[PointURVar],
						 variables[SliderVar],
						 variables[AngleVar]);
   constraints[ConstAngle]->VarChoices(Scheme1, 3, 3, 3, 3, 3);
   constraints[ConstAngle]->VarChoices(Scheme2, 3, 3, 3, 3, 3);
   constraints[ConstAngle]->VarChoices(Scheme3, 4, 4, 4, 4, 4);
   constraints[ConstAngle]->VarChoices(Scheme4, 3, 3, 3, 3, 3);
   constraints[ConstAngle]->Priorities(P_Default, P_Default, P_Default,
				       P_Default, P_Highest);
   constraints[ConstSDistVar] = new DistanceConstraint("ConstSDistVar",
						       NumSchemes,
						       variables[CenterVar],
						       variables[SliderVar],
						       variables[SDistVar]);
   constraints[ConstSDistVar]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstSDistVar]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstSDistVar]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstSDistVar]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstSDistVar]->Priorities(P_Lowest, P_HighMedium, P_Default);
   constraints[ConstCDist] = new DistanceConstraint("ConstCDist",
						    NumSchemes,
						    variables[PointDLVar],
						    variables[CenterVar],
						    variables[SDistVar]);
   constraints[ConstCDist]->VarChoices(Scheme1, 2, 2, 2);
   constraints[ConstCDist]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstCDist]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstCDist]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstCDist]->Priorities(P_Default, P_Default, P_Highest);
   constraints[ConstMidpoint] = new MidpointConstraint("ConstMidpoint",
						       NumSchemes,
						       variables[PointULVar],
						       variables[PointDRVar],
						       variables[CenterVar]);
   constraints[ConstMidpoint]->VarChoices(Scheme1, 2, 2, 2);
   constraints[ConstMidpoint]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstMidpoint]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstMidpoint]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstMidpoint]->Priorities(P_Default, P_Default, P_Highest);
   constraints[ConstPlane] = new PlaneConstraint("ConstPlane",
						 NumSchemes,
						 variables[PointULVar],
						 variables[PointURVar],
						 variables[PointDRVar],
						 variables[PointDLVar]);
   constraints[ConstPlane]->VarChoices(Scheme1, 2, 3, 0, 1);
   constraints[ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 1);
   constraints[ConstPlane]->VarChoices(Scheme3, 2, 3, 0, 1);
   constraints[ConstPlane]->VarChoices(Scheme4, 2, 3, 0, 1);
   constraints[ConstPlane]->Priorities(P_Highest, P_Highest,
				       P_Highest, P_Highest);
   constraints[ConstSPlane] = new PlaneConstraint("ConstSPlane",
						  NumSchemes,
						  variables[PointDLVar],
						  variables[PointURVar],
						  variables[PointDRVar],
						  variables[SliderVar]);
   constraints[ConstSPlane]->VarChoices(Scheme1, 3, 3, 3, 3);
   constraints[ConstSPlane]->VarChoices(Scheme2, 3, 3, 3, 3);
   constraints[ConstSPlane]->VarChoices(Scheme3, 3, 3, 3, 3);
   constraints[ConstSPlane]->VarChoices(Scheme4, 3, 3, 3, 3);
   constraints[ConstSPlane]->Priorities(P_Highest, P_Highest,
					P_Highest, P_Highest);
   constraints[ConstULDR] = new DistanceConstraint("Const13",
						   NumSchemes,
						   variables[PointULVar],
						   variables[PointDRVar],
						   variables[HypoVar]);
   constraints[ConstULDR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstULDR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[ConstULDR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[ConstURDL] = new DistanceConstraint("Const24",
						   NumSchemes,
						   variables[PointURVar],
						   variables[PointDLVar],
						   variables[HypoVar]);
   constraints[ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstURDL]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstURDL]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[ConstHypo] = new RatioConstraint("ConstRatio",
						NumSchemes,
						variables[HypoVar],
						variables[DistVar],
						variables[ConstVar]);
   constraints[ConstHypo]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstHypo]->Priorities(P_HighMedium, P_Default, P_Lowest);
   constraints[ConstULUR] = new DistanceConstraint("Const12",
						   NumSchemes,
						   variables[PointULVar],
						   variables[PointURVar],
						   variables[DistVar]);
   constraints[ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstULUR]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstULDL] = new DistanceConstraint("Const14",
						   NumSchemes,
						   variables[PointULVar],
						   variables[PointDLVar],
						   variables[DistVar]);
   constraints[ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstULDL]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstDRUR] = new DistanceConstraint("Const32",
						   NumSchemes,
						   variables[PointDRVar],
						   variables[PointURVar],
						   variables[DistVar]);
   constraints[ConstDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstDRUR]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstDRDL] = new DistanceConstraint("Const34",
						   NumSchemes,
						   variables[PointDRVar],
						   variables[PointDLVar],
						   variables[DistVar]);
   constraints[ConstDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstDRDL]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[PointMatl] = PointWidgetMaterial;
   materials[EdgeMatl] = EdgeWidgetMaterial;
   materials[SliderMatl] = SliderWidgetMaterial;
   materials[SpecialMatl] = SpecialWidgetMaterial;
   materials[HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = GeomPointUL, pick = PickSphUL;
	geom <= GeomPointDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[PointMatl]);
   
   geometries[GeomRing] = new GeomTorus;
   picks[PickCyls] = new GeomPick(geometries[GeomRing], module);
   picks[PickCyls]->set_highlight(materials[HighMatl]);
   picks[PickCyls]->set_cbdata((void*)PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[PickCyls], materials[EdgeMatl]);

   geometries[GeomResize] = new GeomCappedCylinder;
   picks[PickResize] = new GeomPick(geometries[GeomResize], module);
   picks[PickResize]->set_highlight(materials[HighMatl]);
   picks[PickResize]->set_cbdata((void*)PickResize);
   GeomMaterial* resizem = new GeomMaterial(picks[PickResize], materials[SpecialMatl]);

   geometries[GeomSlider] = new GeomCappedCylinder;
   picks[PickSlider] = new GeomPick(geometries[GeomSlider], module);
   picks[PickSlider]->set_highlight(materials[HighMatl]);
   picks[PickSlider]->set_cbdata((void*)PickSlider);
   GeomMaterial* slidersm = new GeomMaterial(picks[PickSlider], materials[SliderMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(cylsm);
   w->add(resizem);
   w->add(slidersm);

   SetEpsilon(widget_scale*1e-6);
   
   FinishWidget(w);
}


RingWidget::~RingWidget()
{
}


void
RingWidget::widget_execute()
{
   ((GeomSphere*)geometries[GeomPointUL])->move(variables[PointULVar]->point(),
						   1*widget_scale);
   ((GeomSphere*)geometries[GeomPointUR])->move(variables[PointURVar]->point(),
						   1*widget_scale);
   ((GeomSphere*)geometries[GeomPointDR])->move(variables[PointDRVar]->point(),
						   1*widget_scale);
   ((GeomSphere*)geometries[GeomPointDL])->move(variables[PointDLVar]->point(),
						   1*widget_scale);
   Vector normal(Plane(variables[PointULVar]->point(),
		       variables[PointURVar]->point(),
		       variables[PointDLVar]->point()).normal());
   Real rad = (variables[PointULVar]->point()-variables[PointDRVar]->point()).length()/2.;
   ((GeomTorus*)geometries[GeomRing])->move(variables[CenterVar]->point(), normal,
					    rad, 0.5*widget_scale);
   Vector v = variables[SliderVar]->point()-variables[CenterVar]->point();
   Vector slide;
   if (v.length2() > 1e-6)
      slide = Cross(normal, v.normal());
   else
      slide = Vector(0,0,0);
   ((GeomCappedCylinder*)geometries[GeomSlider])->move(variables[SliderVar]->point()
						       - (slide * 0.3 * widget_scale),
						       variables[SliderVar]->point()
						       + (slide * 0.3 * widget_scale),
						       1.1*widget_scale);
   v = variables[PointDRVar]->point()-variables[CenterVar]->point();
   Vector resize;
   if (v.length2() > 1e-6)
      resize = v.normal();
   else
      resize = Vector(0,0,0);
   ((GeomCappedCylinder*)geometries[GeomResize])->move(variables[PointDRVar]->point(),
						       variables[PointDRVar]->point()
						       + (resize * 1.5 * widget_scale),
						       0.5*widget_scale);
   
   ((DistanceConstraint*)constraints[ConstULUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ConstDRDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ConstULDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ConstDRUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ConstULDR])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);
   ((DistanceConstraint*)constraints[ConstURDL])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);

   SetEpsilon(widget_scale*1e-6);

   Vector spvec1(variables[PointURVar]->point() - variables[PointULVar]->point());
   Vector spvec2(variables[PointDLVar]->point() - variables[PointULVar]->point());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == PickSlider)
	    picks[geom]->set_principal(slide);
	 else if (geom == PickResize)
	    picks[geom]->set_principal(resize);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}


void
RingWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			void* cbdata )
{
   ((DistanceConstraint*)constraints[ConstULUR])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[ConstDRDL])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[ConstULDL])->SetDefault(GetAxis2());
   ((DistanceConstraint*)constraints[ConstDRUR])->SetDefault(GetAxis2());

   switch((int)cbdata){
   case PickSphUL:
      variables[PointULVar]->SetDelta(delta);
      break;
   case PickSphUR:
      variables[PointURVar]->SetDelta(delta);
      break;
   case PickSphDR:
      variables[PointDRVar]->SetDelta(delta);
      break;
   case PickSphDL:
      variables[PointDLVar]->SetDelta(delta);
      break;
   case PickResize:
      variables[PointULVar]->MoveDelta(-delta);
      variables[PointDRVar]->SetDelta(delta, Scheme4);
      break;
   case PickSlider:
      variables[SliderVar]->SetDelta(delta);
      break;
   case PickCyls:
      MoveDelta(delta);
      break;
   }
}


void
RingWidget::MoveDelta( const Vector& delta )
{
   variables[PointULVar]->MoveDelta(delta);
   variables[PointURVar]->MoveDelta(delta);
   variables[PointDRVar]->MoveDelta(delta);
   variables[PointDLVar]->MoveDelta(delta);
   variables[CenterVar]->MoveDelta(delta);
   variables[SliderVar]->MoveDelta(delta);
}


Point
RingWidget::ReferencePoint() const
{
   return variables[CenterVar]->point();
}


void
RingWidget::SetPosition( const Point& center, const Vector& normal, const Real radius )
{
   NOT_FINISHED("SetPosition");
   Vector v1, v2;
   normal.find_orthogonal(v1, v2);
   variables[CenterVar]->Move(center);
   variables[PointULVar]->Move(center+v1*radius);
   variables[PointURVar]->Move(center+v2*radius);
   variables[PointDRVar]->Move(center-v1*radius);
   variables[PointDLVar]->Set(center-v2*radius);
   execute();
}


void
RingWidget::GetPosition( Point& center, Vector& normal, Real& radius ) const
{
   center = variables[CenterVar]->point();
   normal = Plane(variables[PointULVar]->point(),
		  variables[PointURVar]->point(),
		  variables[PointDLVar]->point()).normal();
   radius = variables[SDistVar]->real();
}


void
RingWidget::SetRatio( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[AngleVar]->Set(ratio);
   execute();
}


Real
RingWidget::GetRatio() const
{
   return (variables[AngleVar]->real() + 3.14159) / (2.0 * 3.14159);
}


void
RingWidget::SetRadius( const Real radius )
{
   ASSERT(radius>=0.0);
   
   Vector axis1(variables[PointULVar]->point() - variables[CenterVar]->point());
   Vector axis2(variables[PointURVar]->point() - variables[CenterVar]->point());
   Real ratio(radius/variables[SDistVar]->real());

   variables[PointULVar]->Move(variables[CenterVar]->point()+axis1*ratio);
   variables[PointDRVar]->Move(variables[CenterVar]->point()-axis1*ratio);
   variables[PointURVar]->Move(variables[CenterVar]->point()+axis2*ratio);
   variables[PointDLVar]->Move(variables[CenterVar]->point()-axis2*ratio);

   variables[DistVar]->Move(variables[DistVar]->point()*ratio);
   variables[HypoVar]->Move(variables[HypoVar]->point()*ratio);
   
   variables[SDistVar]->Set(radius); // This should set the slider...

   execute();
}

Real
RingWidget::GetRadius() const
{
   return variables[SDistVar]->real();
}


const Vector&
RingWidget::GetAxis1()
{
   Vector axis(variables[PointURVar]->point() - variables[PointULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


const Vector&
RingWidget::GetAxis2()
{
   Vector axis(variables[PointDLVar]->point() - variables[PointULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


