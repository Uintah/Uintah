
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
#include <Constraints/DistanceConstraint.h>
#include <Constraints/HypotenuseConstraint.h>
#include <Constraints/PlaneConstraint.h>
#include <Constraints/SegmentConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>
#include <Geom/Torus.h>
#include <Geometry/Plane.h>

const Index NumCons = 11;
const Index NumVars = 9;
const Index NumGeoms = 7;
const Index NumMatls = 4;
const Index NumPcks = 6;
const Index NumSchemes = 3;

enum { RingW_ConstULDR, RingW_ConstURDL, RingW_ConstHypo, RingW_ConstPlane,
       RingW_ConstULUR, RingW_ConstULDL, RingW_ConstDRUR, RingW_ConstDRDL,
       RingW_ConstLine, RingW_ConstSDist, RingW_ConstRatio };
enum { RingW_SphereUL, RingW_SphereUR, RingW_SphereDR, RingW_SphereDL,
       RingW_Cylinder, RingW_SliderCyl, RingW_Ring };
enum { RingW_PointMatl, RingW_EdgeMatl, RingW_SliderMatl, RingW_HighMatl };
enum { RingW_PickSphUL, RingW_PickSphUR, RingW_PickSphDR, RingW_PickSphDL, RingW_PickCyls,
       RingW_PickSlider };

RingWidget::RingWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1)
{
   Real INIT = 1.0*widget_scale;
   variables[RingW_PointUL] = new Variable("PntUL", Scheme1, Point(0, 0, 0));
   variables[RingW_PointUR] = new Variable("PntUR", Scheme2, Point(INIT, 0, 0));
   variables[RingW_PointDR] = new Variable("PntDR", Scheme1, Point(INIT, INIT, 0));
   variables[RingW_PointDL] = new Variable("PntDL", Scheme2, Point(0, INIT, 0));
   variables[RingW_Slider] = new Variable("Slider", Scheme3, Point(INIT/2.0, INIT/2.0, 0));
   variables[RingW_Dist] = new Variable("Dist", Scheme1, Point(INIT, 0, 0));
   variables[RingW_Hypo] = new Variable("Hypo", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));
   variables[RingW_SDist] = new Variable("SDist", Scheme3, Point(sqrt(INIT*INIT/2.0), 0, 0));
   variables[RingW_Ratio] = new Variable("Ratio", Scheme1, Point(0.5, 0, 0));

   constraints[RingW_ConstLine] = new SegmentConstraint("ConstLine",
							NumSchemes,
							variables[RingW_PointUL],
							variables[RingW_PointDR],
							variables[RingW_Slider]);
   constraints[RingW_ConstLine]->VarChoices(Scheme1, 2, 2, 2);
   constraints[RingW_ConstLine]->VarChoices(Scheme2, 2, 2, 2);
   constraints[RingW_ConstLine]->VarChoices(Scheme3, 2, 2, 2);
   constraints[RingW_ConstLine]->Priorities(P_Default, P_Default, P_Highest);
   constraints[RingW_ConstSDist] = new DistanceConstraint("ConstSDist",
							  NumSchemes,
							  variables[RingW_PointUL],
							  variables[RingW_Slider],
							  variables[RingW_SDist]);
   constraints[RingW_ConstSDist]->VarChoices(Scheme1, 1, 1, 1);
   constraints[RingW_ConstSDist]->VarChoices(Scheme2, 1, 1, 1);
   constraints[RingW_ConstSDist]->VarChoices(Scheme3, 2, 2, 2);
   constraints[RingW_ConstSDist]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[RingW_ConstRatio] = new RatioConstraint("ConstRatio",
						       NumSchemes,
						       variables[RingW_SDist],
						       variables[RingW_Dist],
						       variables[RingW_Ratio]);
   constraints[RingW_ConstRatio]->VarChoices(Scheme1, 0, 0, 0);
   constraints[RingW_ConstRatio]->VarChoices(Scheme2, 0, 0, 0);
   constraints[RingW_ConstRatio]->VarChoices(Scheme3, 2, 2, 2);
   constraints[RingW_ConstRatio]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[RingW_ConstPlane] = new PlaneConstraint("ConstPlane",
							  NumSchemes,
							  variables[RingW_PointUL],
							  variables[RingW_PointUR],
							  variables[RingW_PointDR],
							  variables[RingW_PointDL]);
   
   constraints[RingW_ConstPlane]->VarChoices(Scheme1, 2, 3, 0, 1);
   constraints[RingW_ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 1);
   constraints[RingW_ConstPlane]->VarChoices(Scheme3, 2, 3, 0, 1);
   constraints[RingW_ConstPlane]->Priorities(P_Highest, P_Highest,
					     P_Highest, P_Highest);
   constraints[RingW_ConstULDR] = new DistanceConstraint("Const13",
							    NumSchemes,
							    variables[RingW_PointUL],
							    variables[RingW_PointDR],
							    variables[RingW_Hypo]);
   constraints[RingW_ConstULDR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[RingW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[RingW_ConstULDR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[RingW_ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[RingW_ConstURDL] = new DistanceConstraint("Const24",
							 NumSchemes,
							 variables[RingW_PointUR],
							 variables[RingW_PointDL],
							 variables[RingW_Hypo]);
   constraints[RingW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[RingW_ConstURDL]->VarChoices(Scheme2, 2, 2, 1);
   constraints[RingW_ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[RingW_ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[RingW_ConstHypo] = new HypotenuseConstraint("ConstHypo",
							   NumSchemes,
							   variables[RingW_Dist],
							   variables[RingW_Hypo]);
   constraints[RingW_ConstHypo]->VarChoices(Scheme1, 1, 0);
   constraints[RingW_ConstHypo]->VarChoices(Scheme2, 1, 0);
   constraints[RingW_ConstHypo]->VarChoices(Scheme3, 1, 0);
   constraints[RingW_ConstHypo]->Priorities(P_Default, P_HighMedium);
   constraints[RingW_ConstULUR] = new DistanceConstraint("Const12",
							 NumSchemes,
							 variables[RingW_PointUL],
							 variables[RingW_PointUR],
							 variables[RingW_Dist]);
   constraints[RingW_ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[RingW_ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[RingW_ConstULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[RingW_ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[RingW_ConstULDL] = new DistanceConstraint("Const14",
							 NumSchemes,
							 variables[RingW_PointUL],
							 variables[RingW_PointDL],
							 variables[RingW_Dist]);
   constraints[RingW_ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[RingW_ConstULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[RingW_ConstULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[RingW_ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[RingW_ConstDRUR] = new DistanceConstraint("Const32",
							 NumSchemes,
							 variables[RingW_PointDR],
							 variables[RingW_PointUR],
							 variables[RingW_Dist]);
   constraints[RingW_ConstDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[RingW_ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[RingW_ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[RingW_ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[RingW_ConstDRDL] = new DistanceConstraint("Const34",
							 NumSchemes,
							 variables[RingW_PointDR],
							 variables[RingW_PointDL],
							 variables[RingW_Dist]);
   constraints[RingW_ConstDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[RingW_ConstDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[RingW_ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[RingW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[RingW_PointMatl] = PointWidgetMaterial;
   materials[RingW_EdgeMatl] = EdgeWidgetMaterial;
   materials[RingW_SliderMatl] = SliderWidgetMaterial;
   materials[RingW_HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = RingW_SphereUL, pick = RingW_PickSphUL;
	geom <= RingW_SphereDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[RingW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[RingW_PointMatl]);
   
   GeomGroup* cyls = new GeomGroup;
   geometries[RingW_Cylinder] = new GeomCylinder;
   cyls->add(geometries[RingW_Cylinder]);
   geometries[RingW_Ring] = new GeomTorus;
   cyls->add(geometries[RingW_Ring]);
   picks[RingW_PickCyls] = new GeomPick(cyls, module);
   picks[RingW_PickCyls]->set_highlight(materials[RingW_HighMatl]);
   picks[RingW_PickCyls]->set_cbdata((void*)RingW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[RingW_PickCyls], materials[RingW_EdgeMatl]);

   geometries[RingW_SliderCyl] = new GeomCylinder;
   picks[RingW_PickSlider] = new GeomPick(geometries[RingW_SliderCyl], module);
   picks[RingW_PickSlider]->set_highlight(materials[RingW_HighMatl]);
   picks[RingW_PickSlider]->set_cbdata((void*)RingW_PickSlider);
   GeomMaterial* slidersm = new GeomMaterial(picks[RingW_PickSlider], materials[RingW_SliderMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(cylsm);
   w->add(slidersm);

   SetEpsilon(widget_scale*1e-4);
   
   FinishWidget(w);
}


RingWidget::~RingWidget()
{
}


void
RingWidget::widget_execute()
{
   ((GeomSphere*)geometries[RingW_SphereUL])->move(variables[RingW_PointUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[RingW_SphereUR])->move(variables[RingW_PointUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[RingW_SphereDR])->move(variables[RingW_PointDR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[RingW_SphereDL])->move(variables[RingW_PointDL]->Get(),
						    1*widget_scale);
   ((GeomCylinder*)geometries[RingW_Cylinder])->move(variables[RingW_PointUL]->Get(),
						     variables[RingW_PointDR]->Get(),
						     0.5*widget_scale);
   Point cen=AffineCombination(variables[RingW_PointUL]->Get(), 0.25,
			       variables[RingW_PointUR]->Get(), 0.25,
			       variables[RingW_PointDL]->Get(), 0.25,
			       variables[RingW_PointDR]->Get(), 0.25);
   Vector normal(Plane(variables[RingW_PointUL]->Get(),
		       variables[RingW_PointUR]->Get(),
		       variables[RingW_PointDL]->Get()).normal());
   double rad=(variables[RingW_PointUL]->Get()-variables[RingW_PointDR]->Get()).length()/2.;

   ((GeomTorus*)geometries[RingW_Ring])->move(cen, normal,
					      rad, 0.5*widget_scale);
   ((GeomCylinder*)geometries[RingW_SliderCyl])->move(variables[RingW_Slider]->Get()
							- (GetAxis() * 0.3 * widget_scale),
							variables[RingW_Slider]->Get()
							+ (GetAxis() * 0.3 * widget_scale),
							1.1*widget_scale);

   SetEpsilon(widget_scale*1e-4);

   Vector spvec1(variables[RingW_PointUR]->Get() - variables[RingW_PointUL]->Get());
   Vector spvec2(variables[RingW_PointDL]->Get() - variables[RingW_PointUL]->Get());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == RingW_PickSlider)
	    picks[geom]->set_principal(GetAxis());
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
RingWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   ((DistanceConstraint*)constraints[RingW_ConstSDist])->SetDefault(GetAxis());

   switch((int)cbdata){
   case RingW_PickSphUL:
      variables[RingW_PointUL]->SetDelta(delta);
      break;
   case RingW_PickSphUR:
      variables[RingW_PointUR]->SetDelta(delta);
      break;
   case RingW_PickSphDR:
      variables[RingW_PointDR]->SetDelta(delta);
      break;
   case RingW_PickSphDL:
      variables[RingW_PointDL]->SetDelta(delta);
      break;
   case RingW_PickSlider:
      variables[RingW_Slider]->SetDelta(delta);
      break;
   case RingW_PickCyls:
      variables[RingW_PointUL]->MoveDelta(delta);
      variables[RingW_PointUR]->MoveDelta(delta);
      variables[RingW_PointDR]->MoveDelta(delta);
      variables[RingW_PointDL]->MoveDelta(delta);
      variables[RingW_Slider]->MoveDelta(delta);
      break;
   }
}

