
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
#include <Constraints/CenterConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>
#include <Geom/Torus.h>
#include <Geometry/Plane.h>

const Index NumCons = 7;
const Index NumVars = 8;
const Index NumGeoms = 8;
const Index NumPcks = 8;
const Index NumMdes = 4;
const Index NumSwtchs = 3;
const Index NumSchemes = 5;

enum { ConstAB, ConstBC, ConstCA, ConstCenter,
       ConstRadius, ConstCDist, ConstAngle };
enum { GeomPointA, GeomPointB, GeomPointC,
       GeomSlider, GeomRing, GeomResizeA, GeomResizeB, GeomResizeC };
enum { PickSphA, PickSphB, PickSphC, PickCyls,
       PickSlider, PickResizeA, PickResizeB, PickResizeC };

RingWidget::RingWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumPcks, NumMdes, NumSwtchs, widget_scale*0.1),
  oldaxis1(1, 0, 0), oldaxis2(-0.5, -sqrt(0.75), 0), oldaxis3(-0.5, sqrt(0.75), 0)
{
   Real INIT = widget_scale*2.0/3.0;
   // Scheme5 is used to resize.
   variables[PointAVar] = new PointVariable("PntA", solve, Scheme1, Point(INIT, 0, 0));
   variables[PointBVar] = new PointVariable("PntB", solve, Scheme2, Point(-INIT*0.5, -INIT*sqrt(0.75), 0));
   variables[PointCVar] = new PointVariable("PntC", solve, Scheme3, Point(-INIT*0.5, INIT*sqrt(0.75), 0));
   variables[CenterVar] = new PointVariable("Center", solve, Scheme1, Point(0, 0, 0));
   variables[SliderVar] = new PointVariable("Slider", solve, Scheme4, Point(INIT, 0, 0));
   variables[DistVar] = new RealVariable("Dist", solve, Scheme5, INIT*sqrt(3.0));
   variables[RadiusVar] = new RealVariable("SDist", solve, Scheme5, INIT);
   variables[AngleVar] = new RealVariable("Angle", solve, Scheme4, 0);

   constraints[ConstAngle] = new AngleConstraint("ConstAngle",
						 NumSchemes,
						 variables[CenterVar],
						 variables[PointAVar],
						 variables[PointBVar],
						 variables[SliderVar],
						 variables[AngleVar]);
   constraints[ConstAngle]->VarChoices(Scheme1, 3, 3, 3, 3, 3);
   constraints[ConstAngle]->VarChoices(Scheme2, 3, 3, 3, 3, 3);
   constraints[ConstAngle]->VarChoices(Scheme3, 3, 3, 3, 3, 3);
   constraints[ConstAngle]->VarChoices(Scheme4, 4, 4, 4, 4, 4);
   constraints[ConstAngle]->VarChoices(Scheme5, 3, 3, 3, 3, 3);
   constraints[ConstAngle]->Priorities(P_Default, P_Default, P_Default,
				       P_Default, P_Highest);
   constraints[ConstRadius] = new DistanceConstraint("ConstRadius",
						     NumSchemes,
						     variables[CenterVar],
						     variables[SliderVar],
						     variables[RadiusVar]);
   constraints[ConstRadius]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstRadius]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstRadius]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstRadius]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstRadius]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstRadius]->Priorities(P_Lowest, P_HighMedium, P_Default);
   constraints[ConstCDist] = new DistanceConstraint("ConstCDist",
						    NumSchemes,
						    variables[PointAVar],
						    variables[CenterVar],
						    variables[RadiusVar]);
   constraints[ConstCDist]->VarChoices(Scheme1, 2, 2, 2);
   constraints[ConstCDist]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstCDist]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstCDist]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstCDist]->VarChoices(Scheme5, 2, 2, 2);
   constraints[ConstCDist]->Priorities(P_Default, P_Default, P_Highest);
   constraints[ConstCenter] = new CenterConstraint("ConstCenter",
						   NumSchemes,
						   variables[CenterVar],
						   variables[PointAVar],
						   variables[PointBVar],
						   variables[PointCVar]);
   constraints[ConstCenter]->VarChoices(Scheme1, 0, 0, 0, 0);
   constraints[ConstCenter]->VarChoices(Scheme2, 0, 0, 0, 0);
   constraints[ConstCenter]->VarChoices(Scheme3, 0, 0, 0, 0);
   constraints[ConstCenter]->VarChoices(Scheme4, 0, 0, 0, 0);
   constraints[ConstCenter]->VarChoices(Scheme5, 0, 0, 0, 0);
   constraints[ConstCenter]->Priorities(P_Highest, P_Default, P_Default, P_Default);
   constraints[ConstAB] = new DistanceConstraint("ConstAB",
						 NumSchemes,
						 variables[PointAVar],
						 variables[PointBVar],
						 variables[DistVar]);
   constraints[ConstAB]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstAB]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstAB]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstAB]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstAB]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstAB]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstBC] = new DistanceConstraint("ConstBC",
						 NumSchemes,
						 variables[PointBVar],
						 variables[PointCVar],
						 variables[DistVar]);
   constraints[ConstBC]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstBC]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstBC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstBC]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstBC]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstBC]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstCA] = new DistanceConstraint("ConstCA",
						 NumSchemes,
						 variables[PointCVar],
						 variables[PointAVar],
						 variables[DistVar]);
   constraints[ConstCA]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstCA]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstCA]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstCA]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstCA]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstCA]->Priorities(P_Default, P_Default, P_LowMedium);

   Index geom, pick;
   geometries[GeomRing] = new GeomTorus;
   picks[PickCyls] = new GeomPick(geometries[GeomRing], module);
   picks[PickCyls]->set_highlight(HighlightMaterial);
   picks[PickCyls]->set_cbdata((void*)PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[PickCyls], EdgeMaterial);
   CreateModeSwitch(0, cylsm);

   GeomGroup* pts = new GeomGroup;
   for (geom = GeomPointA, pick = PickSphA;
	geom <= GeomPointC; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(HighlightMaterial);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, PointMaterial);
   
   geometries[GeomResizeA] = new GeomCappedCylinder;
   picks[PickResizeA] = new GeomPick(geometries[GeomResizeA], module);
   picks[PickResizeA]->set_highlight(HighlightMaterial);
   picks[PickResizeA]->set_cbdata((void*)PickResizeA);
   GeomMaterial* resizeam = new GeomMaterial(picks[PickResizeA], SpecialMaterial);

   GeomGroup* resbc = new GeomGroup;
   geometries[GeomResizeB] = new GeomCappedCylinder;
   picks[PickResizeB] = new GeomPick(geometries[GeomResizeB], module);
   picks[PickResizeB]->set_highlight(HighlightMaterial);
   picks[PickResizeB]->set_cbdata((void*)PickResizeB);
   resbc->add(picks[PickResizeB]);
   geometries[GeomResizeC] = new GeomCappedCylinder;
   picks[PickResizeC] = new GeomPick(geometries[GeomResizeC], module);
   picks[PickResizeC]->set_highlight(HighlightMaterial);
   picks[PickResizeC]->set_cbdata((void*)PickResizeC);
   resbc->add(picks[PickResizeC]);
   GeomMaterial* resizebcm = new GeomMaterial(resbc, ResizeMaterial);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(resizeam);
   w->add(resizebcm);
   CreateModeSwitch(1, w);

   geometries[GeomSlider] = new GeomCappedCylinder;
   picks[PickSlider] = new GeomPick(geometries[GeomSlider], module);
   picks[PickSlider]->set_highlight(HighlightMaterial);
   picks[PickSlider]->set_cbdata((void*)PickSlider);
   GeomMaterial* slidersm = new GeomMaterial(picks[PickSlider], SliderMaterial);
   CreateModeSwitch(2, slidersm);

   SetMode(Mode1, Switch0|Switch1|Switch2);
   SetMode(Mode2, Switch0|Switch1);
   SetMode(Mode3, Switch0|Switch2);
   SetMode(Mode4, Switch0);

   SetEpsilon(widget_scale*1e-6);
   
   FinishWidget();
}


RingWidget::~RingWidget()
{
}


void
RingWidget::widget_execute()
{
   ((GeomSphere*)geometries[GeomPointA])->move(variables[PointAVar]->point(),
					       1*widget_scale);
   ((GeomSphere*)geometries[GeomPointB])->move(variables[PointBVar]->point(),
					       1*widget_scale);
   ((GeomSphere*)geometries[GeomPointC])->move(variables[PointCVar]->point(),
					       1*widget_scale);
   Vector normal(Cross(GetAxis1(), GetAxis2()));
   if (normal.length2() < 1e-6) normal = Vector(1.0, 0.0, 0.0);
   ((GeomTorus*)geometries[GeomRing])->move(variables[CenterVar]->point(), normal,
					    variables[RadiusVar]->real(), 0.5*widget_scale);
   Vector v(variables[SliderVar]->point()-variables[CenterVar]->point());
   Vector slide;
   if (v.length2() > 1e-6) {
      slide = Cross(normal, v.normal());
      if (slide.length2() < 1e-6) slide = Vector(0.0, 1.0, 0.0);
   } else {
      slide = GetAxis1();
   }
   ((GeomCappedCylinder*)geometries[GeomSlider])->move(variables[SliderVar]->point()
						       - (slide * 0.3 * widget_scale),
						       variables[SliderVar]->point()
						       + (slide * 0.3 * widget_scale),
						       1.1*widget_scale);
   ((GeomCappedCylinder*)geometries[GeomResizeA])->move(variables[PointAVar]->point(),
							variables[PointAVar]->point()
							+ (GetAxis1() * 1.5 * widget_scale),
							0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[GeomResizeB])->move(variables[PointBVar]->point(),
							variables[PointBVar]->point()
							+ (GetAxis2() * 1.5 * widget_scale),
							0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[GeomResizeC])->move(variables[PointCVar]->point(),
							variables[PointCVar]->point()
							+ (GetAxis3() * 1.5 * widget_scale),
							0.5*widget_scale);
   
   ((DistanceConstraint*)constraints[ConstAB])->SetMinimum(1.0*widget_scale);
   ((DistanceConstraint*)constraints[ConstBC])->SetMinimum(1.0*widget_scale);
   ((DistanceConstraint*)constraints[ConstCA])->SetMinimum(1.0*widget_scale);

   SetEpsilon(widget_scale*1e-6);
   
   Vector spvec1(GetAxis1());
   Vector spvec2(variables[PointCVar]->point() - variables[PointBVar]->point());
   if (spvec2.length2() > 0.0) {
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == PickSlider)
	    picks[geom]->set_principal(slide);
	 else if (geom == PickResizeA)
	    picks[geom]->set_principal(GetAxis1());
	 else if (geom == PickResizeB)
	    picks[geom]->set_principal(GetAxis2());
	 else if (geom == PickResizeC)
	    picks[geom]->set_principal(GetAxis3());
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}


void
RingWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			void* cbdata )
{
   Vector delt(delta);
   ((DistanceConstraint*)constraints[ConstAB])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[ConstBC])->SetDefault(GetAxis2());
   ((DistanceConstraint*)constraints[ConstCA])->SetDefault(GetAxis3());

   switch((int)cbdata){
   case PickSphA:
      variables[PointAVar]->SetDelta(delta);
      break;
   case PickSphB:
      variables[PointBVar]->SetDelta(delta);
      break;
   case PickSphC:
      variables[PointCVar]->SetDelta(delta);
      break;
   case PickResizeA:
      if (((variables[PointAVar]->point()+delta)-variables[CenterVar]->point()).length()
	  < 1.0*widget_scale) {
	 delt = ((variables[CenterVar]->point() + delta.normal()*1.0*widget_scale)
		 - variables[PointAVar]->point());
      }
      SetRadius(((variables[PointAVar]->point()+delt)-variables[CenterVar]->point()).length());
      break;
   case PickResizeB:
      if (((variables[PointBVar]->point()+delta)-variables[CenterVar]->point()).length()
	  < 1.0*widget_scale) {
	 delt = ((variables[CenterVar]->point() + delta.normal()*1.0*widget_scale)
		 - variables[PointBVar]->point());
      }
      SetRadius(((variables[PointBVar]->point()+delt)-variables[CenterVar]->point()).length());
      break;
   case PickResizeC:
      if (((variables[PointCVar]->point()+delta)-variables[CenterVar]->point()).length()
	  < 1.0*widget_scale) {
	 delt = ((variables[CenterVar]->point() + delta.normal()*1.0*widget_scale)
		 - variables[PointCVar]->point());
      }
      SetRadius(((variables[PointCVar]->point()+delt)-variables[CenterVar]->point()).length());
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
   variables[PointAVar]->MoveDelta(delta);
   variables[PointBVar]->MoveDelta(delta);
   variables[PointCVar]->MoveDelta(delta);
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
   NOT_FINISHED("SetPosition");
   Vector v1, v2;
   normal.find_orthogonal(v1, v2);
   variables[CenterVar]->Move(center);
   variables[PointAVar]->Move(center+v1*radius);
   variables[PointBVar]->Move(center-v1*radius*0.5-v2*radius*sqrt(0.75));
   variables[PointCVar]->Set(center-v1*radius*0.5+v2*radius*sqrt(0.75));

   execute();
}


void
RingWidget::GetPosition( Point& center, Vector& normal, Real& radius ) const
{
   center = variables[CenterVar]->point();
   normal = Plane(variables[PointAVar]->point(),
		  variables[PointBVar]->point(),
		  variables[PointCVar]->point()).normal();
   radius = variables[RadiusVar]->real();
}


void
RingWidget::SetRatio( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[AngleVar]->Set(ratio*2.0*3.14159 - 3.14159);

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
   
   Vector axis1(variables[PointAVar]->point() - variables[CenterVar]->point());
   Vector axis2(variables[PointBVar]->point() - variables[CenterVar]->point());
   Vector axis3(variables[PointCVar]->point() - variables[CenterVar]->point());
   Real ratio(radius/variables[RadiusVar]->real());

   variables[PointAVar]->Move(variables[CenterVar]->point()+axis1*ratio);
   variables[PointBVar]->Move(variables[CenterVar]->point()+axis2*ratio);
   variables[PointCVar]->Move(variables[CenterVar]->point()+axis3*ratio);

   variables[DistVar]->Move(variables[DistVar]->real()*ratio);
   
   variables[RadiusVar]->Set(radius); // This should set the slider...

   execute();
}


Real
RingWidget::GetRadius() const
{
   return variables[RadiusVar]->real();
}


const Vector&
RingWidget::GetAxis1()
{
   Vector axis(variables[PointAVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


const Vector&
RingWidget::GetAxis2()
{
   Vector axis(variables[PointBVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


const Vector&
RingWidget::GetAxis3()
{
   Vector axis(variables[PointCVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis3;
   else
      return (oldaxis3 = axis.normal());
}


