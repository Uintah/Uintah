
/*
 *  ScaledSquareWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/ScaledSquareWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/HypotenuseConstraint.h>
#include <Constraints/PlaneConstraint.h>
#include <Constraints/SegmentConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 14;
const Index NumVars = 12;
const Index NumGeoms = 10;
const Index NumMatls = 4;
const Index NumPcks = 7;
const Index NumSchemes = 4;

enum { SSquareW_ConstULDR, SSquareW_ConstURDL, SSquareW_ConstHypo, SSquareW_ConstPlane,
       SSquareW_ConstULUR, SSquareW_ConstULDL, SSquareW_ConstDRUR, SSquareW_ConstDRDL,
       SSquareW_ConstLine1, SSquareW_ConstSDist1, SSquareW_ConstRatio1,
       SSquareW_ConstLine2, SSquareW_ConstSDist2, SSquareW_ConstRatio2 };
enum { SSquareW_SphereUL, SSquareW_SphereUR, SSquareW_SphereDR, SSquareW_SphereDL,
       SSquareW_CylU, SSquareW_CylR, SSquareW_CylD, SSquareW_CylL,
       SSquareW_SliderCyl1, SSquareW_SliderCyl2 };
enum { SSquareW_PickSphUL, SSquareW_PickSphUR, SSquareW_PickSphDR, SSquareW_PickSphDL, SSquareW_PickCyls,
       SSquareW_PickSlider1, SSquareW_PickSlider2 };

ScaledSquareWidget::ScaledSquareWidget( Module* module, CrowdMonitor* lock,
				       Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1)
{
   Real INIT = 1.0*widget_scale;
   variables[SSquareW_PointUL] = new Variable("PntUL", Scheme1, Point(0, 0, 0));
   variables[SSquareW_PointUR] = new Variable("PntUR", Scheme2, Point(INIT, 0, 0));
   variables[SSquareW_PointDR] = new Variable("PntDR", Scheme1, Point(INIT, INIT, 0));
   variables[SSquareW_PointDL] = new Variable("PntDL", Scheme2, Point(0, INIT, 0));
   variables[SSquareW_Slider1] = new Variable("Slider1", Scheme3, Point(INIT/2.0, 0, 0));
   variables[SSquareW_Slider2] = new Variable("Slider2", Scheme4, Point(0, INIT/2.0, 0));
   variables[SSquareW_Dist] = new Variable("Dist", Scheme1, Point(INIT, 0, 0));
   variables[SSquareW_Hypo] = new Variable("Hypo", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));
   variables[SSquareW_SDist1] = new Variable("SDist1", Scheme3, Point(INIT/2.0, 0, 0));
   variables[SSquareW_SDist2] = new Variable("SDist2", Scheme4, Point(INIT/2.0, 0, 0));
   variables[SSquareW_Ratio1] = new Variable("Ratio1", Scheme1, Point(0.5, 0, 0));
   variables[SSquareW_Ratio2] = new Variable("Ratio2", Scheme1, Point(0.5, 0, 0));

   constraints[SSquareW_ConstLine1] = new SegmentConstraint("ConstLine1",
							 NumSchemes,
							 variables[SSquareW_PointUL],
							 variables[SSquareW_PointUR],
							 variables[SSquareW_Slider1]);
   constraints[SSquareW_ConstLine1]->VarChoices(Scheme1, 2, 2, 2);
   constraints[SSquareW_ConstLine1]->VarChoices(Scheme2, 2, 2, 2);
   constraints[SSquareW_ConstLine1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SSquareW_ConstLine1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SSquareW_ConstLine1]->Priorities(P_Default, P_Default, P_Highest);
   constraints[SSquareW_ConstLine2] = new SegmentConstraint("ConstLine2",
							 NumSchemes,
							 variables[SSquareW_PointUL],
							 variables[SSquareW_PointDL],
							 variables[SSquareW_Slider2]);
   constraints[SSquareW_ConstLine2]->VarChoices(Scheme1, 2, 2, 2);
   constraints[SSquareW_ConstLine2]->VarChoices(Scheme2, 2, 2, 2);
   constraints[SSquareW_ConstLine2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SSquareW_ConstLine2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SSquareW_ConstLine2]->Priorities(P_Default, P_Default, P_Highest);
   constraints[SSquareW_ConstSDist1] = new DistanceConstraint("ConstSDist1",
							  NumSchemes,
							  variables[SSquareW_PointUL],
							  variables[SSquareW_Slider1],
							  variables[SSquareW_SDist1]);
   constraints[SSquareW_ConstSDist1]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SSquareW_ConstSDist1]->VarChoices(Scheme2, 1, 1, 1);
   constraints[SSquareW_ConstSDist1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SSquareW_ConstSDist1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SSquareW_ConstSDist1]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[SSquareW_ConstRatio1] = new RatioConstraint("ConstRatio1",
							NumSchemes,
							variables[SSquareW_SDist1],
							variables[SSquareW_Dist],
							variables[SSquareW_Ratio1]);
   constraints[SSquareW_ConstRatio1]->VarChoices(Scheme1, 0, 0, 0);
   constraints[SSquareW_ConstRatio1]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SSquareW_ConstRatio1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SSquareW_ConstRatio1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SSquareW_ConstRatio1]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[SSquareW_ConstSDist2] = new DistanceConstraint("ConstSDist2",
							  NumSchemes,
							  variables[SSquareW_PointUL],
							  variables[SSquareW_Slider2],
							  variables[SSquareW_SDist2]);
   constraints[SSquareW_ConstSDist2]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SSquareW_ConstSDist2]->VarChoices(Scheme2, 1, 1, 1);
   constraints[SSquareW_ConstSDist2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SSquareW_ConstSDist2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SSquareW_ConstSDist2]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[SSquareW_ConstRatio2] = new RatioConstraint("ConstRatio2",
							NumSchemes,
							variables[SSquareW_SDist2],
							variables[SSquareW_Dist],
							variables[SSquareW_Ratio2]);
   constraints[SSquareW_ConstRatio2]->VarChoices(Scheme1, 0, 0, 0);
   constraints[SSquareW_ConstRatio2]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SSquareW_ConstRatio2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SSquareW_ConstRatio2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SSquareW_ConstRatio2]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[SSquareW_ConstPlane] = new PlaneConstraint("ConstPlane",
							  NumSchemes,
							  variables[SSquareW_PointUL],
							  variables[SSquareW_PointUR],
							  variables[SSquareW_PointDR],
							  variables[SSquareW_PointDL]);
   
   constraints[SSquareW_ConstPlane]->VarChoices(Scheme1, 2, 3, 0, 1);
   constraints[SSquareW_ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 1);
   constraints[SSquareW_ConstPlane]->VarChoices(Scheme3, 2, 3, 0, 1);
   constraints[SSquareW_ConstPlane]->VarChoices(Scheme4, 2, 3, 0, 1);
   constraints[SSquareW_ConstPlane]->Priorities(P_Highest, P_Highest,
					      P_Highest, P_Highest);
   constraints[SSquareW_ConstULDR] = new DistanceConstraint("Const13",
							    NumSchemes,
							    variables[SSquareW_PointUL],
							    variables[SSquareW_PointDR],
							    variables[SSquareW_Hypo]);
   constraints[SSquareW_ConstULDR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[SSquareW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SSquareW_ConstULDR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[SSquareW_ConstULDR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SSquareW_ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[SSquareW_ConstURDL] = new DistanceConstraint("Const24",
							    NumSchemes,
							    variables[SSquareW_PointUR],
							    variables[SSquareW_PointDL],
							    variables[SSquareW_Hypo]);
   constraints[SSquareW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SSquareW_ConstURDL]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SSquareW_ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SSquareW_ConstURDL]->VarChoices(Scheme4, 2, 2, 1);
   constraints[SSquareW_ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[SSquareW_ConstHypo] = new HypotenuseConstraint("ConstHypo",
							      NumSchemes,
							      variables[SSquareW_Dist],
							      variables[SSquareW_Hypo]);
   constraints[SSquareW_ConstHypo]->VarChoices(Scheme1, 1, 0);
   constraints[SSquareW_ConstHypo]->VarChoices(Scheme2, 1, 0);
   constraints[SSquareW_ConstHypo]->VarChoices(Scheme3, 1, 0);
   constraints[SSquareW_ConstHypo]->VarChoices(Scheme4, 1, 0);
   constraints[SSquareW_ConstHypo]->Priorities(P_Default, P_HighMedium);
   constraints[SSquareW_ConstULUR] = new DistanceConstraint("Const12",
							    NumSchemes,
							    variables[SSquareW_PointUL],
							    variables[SSquareW_PointUR],
							    variables[SSquareW_Dist]);
   constraints[SSquareW_ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SSquareW_ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SSquareW_ConstULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SSquareW_ConstULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SSquareW_ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SSquareW_ConstULDL] = new DistanceConstraint("Const14",
							    NumSchemes,
							    variables[SSquareW_PointUL],
							    variables[SSquareW_PointDL],
							    variables[SSquareW_Dist]);
   constraints[SSquareW_ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SSquareW_ConstULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SSquareW_ConstULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SSquareW_ConstULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SSquareW_ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SSquareW_ConstDRUR] = new DistanceConstraint("Const32",
							    NumSchemes,
							    variables[SSquareW_PointDR],
							    variables[SSquareW_PointUR],
							    variables[SSquareW_Dist]);
   constraints[SSquareW_ConstDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SSquareW_ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SSquareW_ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SSquareW_ConstDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SSquareW_ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SSquareW_ConstDRDL] = new DistanceConstraint("Const34",
							  NumSchemes,
							  variables[SSquareW_PointDR],
							  variables[SSquareW_PointDL],
							  variables[SSquareW_Dist]);
   constraints[SSquareW_ConstDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SSquareW_ConstDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SSquareW_ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SSquareW_ConstDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SSquareW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[SSquareW_PointMatl] = PointWidgetMaterial;
   materials[SSquareW_EdgeMatl] = EdgeWidgetMaterial;
   materials[SSquareW_SliderMatl] = SliderWidgetMaterial;
   materials[SSquareW_HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = SSquareW_SphereUL, pick = SSquareW_PickSphUL;
	geom <= SSquareW_SphereDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[SSquareW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[SSquareW_PointMatl]);
   
   GeomGroup* cyls = new GeomGroup;
   for (geom = SSquareW_CylU; geom <= SSquareW_CylL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[SSquareW_PickCyls] = new GeomPick(cyls, module);
   picks[SSquareW_PickCyls]->set_highlight(materials[SSquareW_HighMatl]);
   picks[SSquareW_PickCyls]->set_cbdata((void*)SSquareW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[SSquareW_PickCyls], materials[SSquareW_EdgeMatl]);

   GeomGroup* sliders = new GeomGroup;
   geometries[SSquareW_SliderCyl1] = new GeomCappedCylinder;
   picks[SSquareW_PickSlider1] = new GeomPick(geometries[SSquareW_SliderCyl1], module);
   picks[SSquareW_PickSlider1]->set_highlight(materials[SSquareW_HighMatl]);
   picks[SSquareW_PickSlider1]->set_cbdata((void*)SSquareW_PickSlider1);
   sliders->add(picks[SSquareW_PickSlider1]);
   geometries[SSquareW_SliderCyl2] = new GeomCappedCylinder;
   picks[SSquareW_PickSlider2] = new GeomPick(geometries[SSquareW_SliderCyl2], module);
   picks[SSquareW_PickSlider2]->set_highlight(materials[SSquareW_HighMatl]);
   picks[SSquareW_PickSlider2]->set_cbdata((void*)SSquareW_PickSlider2);
   sliders->add(picks[SSquareW_PickSlider2]);
   GeomMaterial* slidersm = new GeomMaterial(sliders, materials[SSquareW_SliderMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(cylsm);
   w->add(slidersm);
   
   SetEpsilon(widget_scale*1e-4);

   FinishWidget(w);
}


ScaledSquareWidget::~ScaledSquareWidget()
{
}


void
ScaledSquareWidget::widget_execute()
{
   ((GeomSphere*)geometries[SSquareW_SphereUL])->move(variables[SSquareW_PointUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SSquareW_SphereUR])->move(variables[SSquareW_PointUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SSquareW_SphereDR])->move(variables[SSquareW_PointDR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SSquareW_SphereDL])->move(variables[SSquareW_PointDL]->Get(),
						    1*widget_scale);
   ((GeomCylinder*)geometries[SSquareW_CylU])->move(variables[SSquareW_PointUL]->Get(),
						  variables[SSquareW_PointUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SSquareW_CylR])->move(variables[SSquareW_PointUR]->Get(),
						  variables[SSquareW_PointDR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SSquareW_CylD])->move(variables[SSquareW_PointDR]->Get(),
						  variables[SSquareW_PointDL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SSquareW_CylL])->move(variables[SSquareW_PointDL]->Get(),
						  variables[SSquareW_PointUL]->Get(),
						  0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[SSquareW_SliderCyl1])->move(variables[SSquareW_Slider1]->Get()
								- (GetAxis1() * 0.3 * widget_scale),
								variables[SSquareW_Slider1]->Get()
								+ (GetAxis1() * 0.3 * widget_scale),
								1.1*widget_scale);
   ((GeomCappedCylinder*)geometries[SSquareW_SliderCyl2])->move(variables[SSquareW_Slider2]->Get()
								- (GetAxis2() * 0.3 * widget_scale),
								variables[SSquareW_Slider2]->Get()
								+ (GetAxis2() * 0.3 * widget_scale),
								1.1*widget_scale);

   SetEpsilon(widget_scale*1e-4);

   Vector spvec1(variables[SSquareW_PointUR]->Get() - variables[SSquareW_PointUL]->Get());
   Vector spvec2(variables[SSquareW_PointDL]->Get() - variables[SSquareW_PointUL]->Get());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == SSquareW_PickSlider1)
	    picks[geom]->set_principal(spvec1);
	 else if (geom == SSquareW_PickSlider2)
	    picks[geom]->set_principal(spvec2);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
ScaledSquareWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   ((DistanceConstraint*)constraints[SSquareW_ConstSDist1])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[SSquareW_ConstSDist2])->SetDefault(GetAxis2());

   switch((int)cbdata){
   case SSquareW_PickSphUL:
      variables[SSquareW_PointUL]->SetDelta(delta);
      break;
   case SSquareW_PickSphUR:
      variables[SSquareW_PointUR]->SetDelta(delta);
      break;
   case SSquareW_PickSphDR:
      variables[SSquareW_PointDR]->SetDelta(delta);
      break;
   case SSquareW_PickSphDL:
      variables[SSquareW_PointDL]->SetDelta(delta);
      break;
   case SSquareW_PickSlider1:
      variables[SSquareW_Slider1]->SetDelta(delta);
      break;
   case SSquareW_PickSlider2:
      variables[SSquareW_Slider2]->SetDelta(delta);
      break;
   case SSquareW_PickCyls:
      variables[SSquareW_PointUL]->MoveDelta(delta);
      variables[SSquareW_PointUR]->MoveDelta(delta);
      variables[SSquareW_PointDR]->MoveDelta(delta);
      variables[SSquareW_PointDL]->MoveDelta(delta);
      variables[SSquareW_Slider1]->MoveDelta(delta);
      variables[SSquareW_Slider2]->MoveDelta(delta);
      break;
   }
}

