
/*
 *  ScaledFrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/ScaledFrameWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Constraints/PlaneConstraint.h>
#include <Constraints/SegmentConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 14;
const Index NumVars = 13;
const Index NumGeoms = 14;
const Index NumMatls = 4;
const Index NumPcks = 11;
const Index NumSchemes = 6;

enum { SFrameW_ConstULDR, SFrameW_ConstURDL, SFrameW_ConstPyth, SFrameW_ConstPlane,
       SFrameW_ConstULUR, SFrameW_ConstULDL, SFrameW_ConstDRUR, SFrameW_ConstDRDL,
       SFrameW_ConstLine1, SFrameW_ConstSDist1, SFrameW_ConstRatio1,
       SFrameW_ConstLine2, SFrameW_ConstSDist2, SFrameW_ConstRatio2 };
enum { SFrameW_SphereUL, SFrameW_SphereUR, SFrameW_SphereDR, SFrameW_SphereDL,
       SFrameW_CylU, SFrameW_CylR, SFrameW_CylD, SFrameW_CylL,
       SFrameW_GeomResizeU, SFrameW_GeomResizeR, SFrameW_GeomResizeD, SFrameW_GeomResizeL,
       SFrameW_SliderCyl1, SFrameW_SliderCyl2 };
enum { SFrameW_PickSphUL, SFrameW_PickSphUR, SFrameW_PickSphDR, SFrameW_PickSphDL, SFrameW_PickCyls,
       SFrameW_PickResizeU, SFrameW_PickResizeR, SFrameW_PickResizeD, SFrameW_PickResizeL,
       SFrameW_PickSlider1, SFrameW_PickSlider2 };

ScaledFrameWidget::ScaledFrameWidget( Module* module, CrowdMonitor* lock,
				      Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1),
  oldaxis1(1, 0, 0), oldaxis2(0, 1, 0)
{
   Real INIT = 1.0*widget_scale;
   variables[SFrameW_PointUL] = new PointVariable("PntUL", Scheme1, Point(0, 0, 0));
   variables[SFrameW_PointUR] = new PointVariable("PntUR", Scheme2, Point(INIT, 0, 0));
   variables[SFrameW_PointDR] = new PointVariable("PntDR", Scheme1, Point(INIT, INIT, 0));
   variables[SFrameW_PointDL] = new PointVariable("PntDL", Scheme2, Point(0, INIT, 0));
   variables[SFrameW_Slider1] = new PointVariable("Slider1", Scheme3, Point(INIT/2.0, 0, 0));
   variables[SFrameW_Slider2] = new PointVariable("Slider2", Scheme4, Point(0, INIT/2.0, 0));
   variables[SFrameW_Dist1] = new RealVariable("Dist1", Scheme1, INIT);
   variables[SFrameW_Dist2] = new RealVariable("Dist2", Scheme1, INIT);
   variables[SFrameW_Hypo] = new RealVariable("Hypo", Scheme1, sqrt(2*INIT*INIT));
   variables[SFrameW_SDist1] = new RealVariable("SDist1", Scheme3, INIT/2.0);
   variables[SFrameW_SDist2] = new RealVariable("SDist2", Scheme4, INIT/2.0);
   variables[SFrameW_Ratio1] = new RealVariable("Ratio1", Scheme1, 0.5);
   variables[SFrameW_Ratio2] = new RealVariable("Ratio2", Scheme1, 0.5);

   constraints[SFrameW_ConstLine1] = new SegmentConstraint("ConstLine1",
							   NumSchemes,
							   variables[SFrameW_PointUL],
							   variables[SFrameW_PointUR],
							   variables[SFrameW_Slider1]);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme1, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme2, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme5, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme6, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->Priorities(P_Default, P_Default, P_Highest);
   constraints[SFrameW_ConstLine2] = new SegmentConstraint("ConstLine2",
							   NumSchemes,
							   variables[SFrameW_PointUL],
							   variables[SFrameW_PointDL],
							   variables[SFrameW_Slider2]);
   constraints[SFrameW_ConstLine2]->VarChoices(Scheme1, 2, 2, 2);
   constraints[SFrameW_ConstLine2]->VarChoices(Scheme2, 2, 2, 2);
   constraints[SFrameW_ConstLine2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SFrameW_ConstLine2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SFrameW_ConstLine2]->VarChoices(Scheme5, 2, 2, 2);
   constraints[SFrameW_ConstLine2]->VarChoices(Scheme6, 2, 2, 2);
   constraints[SFrameW_ConstLine2]->Priorities(P_Default, P_Default, P_Highest);
   constraints[SFrameW_ConstSDist1] = new DistanceConstraint("ConstSDist1",
							     NumSchemes,
							     variables[SFrameW_PointUL],
							     variables[SFrameW_Slider1],
							     variables[SFrameW_SDist1]);
   constraints[SFrameW_ConstSDist1]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SFrameW_ConstSDist1]->VarChoices(Scheme2, 1, 1, 1);
   constraints[SFrameW_ConstSDist1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SFrameW_ConstSDist1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SFrameW_ConstSDist1]->VarChoices(Scheme5, 1, 1, 1);
   constraints[SFrameW_ConstSDist1]->VarChoices(Scheme6, 1, 1, 1);
   constraints[SFrameW_ConstSDist1]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[SFrameW_ConstRatio1] = new RatioConstraint("ConstRatio1",
							  NumSchemes,
							  variables[SFrameW_SDist1],
							  variables[SFrameW_Dist1],
							  variables[SFrameW_Ratio1]);
   constraints[SFrameW_ConstRatio1]->VarChoices(Scheme1, 0, 0, 0);
   constraints[SFrameW_ConstRatio1]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SFrameW_ConstRatio1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SFrameW_ConstRatio1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SFrameW_ConstRatio1]->VarChoices(Scheme5, 0, 0, 0);
   constraints[SFrameW_ConstRatio1]->VarChoices(Scheme6, 0, 0, 0);
   constraints[SFrameW_ConstRatio1]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[SFrameW_ConstSDist2] = new DistanceConstraint("ConstSDist2",
							     NumSchemes,
							     variables[SFrameW_PointUL],
							     variables[SFrameW_Slider2],
							     variables[SFrameW_SDist2]);
   constraints[SFrameW_ConstSDist2]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SFrameW_ConstSDist2]->VarChoices(Scheme2, 1, 1, 1);
   constraints[SFrameW_ConstSDist2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SFrameW_ConstSDist2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SFrameW_ConstSDist2]->VarChoices(Scheme5, 1, 1, 1);
   constraints[SFrameW_ConstSDist2]->VarChoices(Scheme6, 1, 1, 1);
   constraints[SFrameW_ConstSDist2]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[SFrameW_ConstRatio2] = new RatioConstraint("ConstRatio2",
							  NumSchemes,
							  variables[SFrameW_SDist2],
							  variables[SFrameW_Dist2],
							  variables[SFrameW_Ratio2]);
   constraints[SFrameW_ConstRatio2]->VarChoices(Scheme1, 0, 0, 0);
   constraints[SFrameW_ConstRatio2]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SFrameW_ConstRatio2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SFrameW_ConstRatio2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[SFrameW_ConstRatio2]->VarChoices(Scheme5, 0, 0, 0);
   constraints[SFrameW_ConstRatio2]->VarChoices(Scheme6, 0, 0, 0);
   constraints[SFrameW_ConstRatio2]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[SFrameW_ConstPlane] = new PlaneConstraint("ConstPlane",
							 NumSchemes,
							 variables[SFrameW_PointUL],
							 variables[SFrameW_PointUR],
							 variables[SFrameW_PointDR],
							 variables[SFrameW_PointDL]);
   constraints[SFrameW_ConstPlane]->VarChoices(Scheme1, 2, 3, 0, 1);
   constraints[SFrameW_ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 1);
   constraints[SFrameW_ConstPlane]->VarChoices(Scheme3, 2, 3, 0, 1);
   constraints[SFrameW_ConstPlane]->VarChoices(Scheme4, 2, 3, 0, 1);
   constraints[SFrameW_ConstPlane]->VarChoices(Scheme5, 2, 3, 0, 1);
   constraints[SFrameW_ConstPlane]->VarChoices(Scheme6, 2, 3, 0, 1);
   constraints[SFrameW_ConstPlane]->Priorities(P_Highest, P_Highest,
					       P_Highest, P_Highest);
   constraints[SFrameW_ConstULDR] = new DistanceConstraint("Const13",
							   NumSchemes,
							   variables[SFrameW_PointUL],
							   variables[SFrameW_PointDR],
							   variables[SFrameW_Hypo]);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme5, 2, 2, 1);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme6, 1, 0, 1);
   constraints[SFrameW_ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[SFrameW_ConstURDL] = new DistanceConstraint("Const24",
							   NumSchemes,
							   variables[SFrameW_PointUR],
							   variables[SFrameW_PointDL],
							   variables[SFrameW_Hypo]);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme5, 1, 0, 1);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme6, 2, 2, 1);
   constraints[SFrameW_ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[SFrameW_ConstPyth] = new PythagorasConstraint("ConstPyth",
							     NumSchemes,
							     variables[SFrameW_Dist1],
							     variables[SFrameW_Dist2],
							     variables[SFrameW_Hypo]);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme5, 2, 2, 0);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme6, 2, 2, 1);
   constraints[SFrameW_ConstPyth]->Priorities(P_Default, P_Default, P_HighMedium);
   constraints[SFrameW_ConstULUR] = new DistanceConstraint("Const12",
							   NumSchemes,
							   variables[SFrameW_PointUL],
							   variables[SFrameW_PointUR],
							   variables[SFrameW_Dist1]);
   constraints[SFrameW_ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SFrameW_ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SFrameW_ConstULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SFrameW_ConstULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SFrameW_ConstULUR]->VarChoices(Scheme5, 1, 1, 1);
   constraints[SFrameW_ConstULUR]->VarChoices(Scheme6, 0, 0, 0);
   constraints[SFrameW_ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SFrameW_ConstULDL] = new DistanceConstraint("Const14",
							   NumSchemes,
							   variables[SFrameW_PointUL],
							   variables[SFrameW_PointDL],
							   variables[SFrameW_Dist2]);
   constraints[SFrameW_ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SFrameW_ConstULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SFrameW_ConstULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SFrameW_ConstULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SFrameW_ConstULDL]->VarChoices(Scheme5, 1, 1, 1);
   constraints[SFrameW_ConstULDL]->VarChoices(Scheme6, 0, 0, 0);
   constraints[SFrameW_ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SFrameW_ConstDRUR] = new DistanceConstraint("Const32",
							   NumSchemes,
							   variables[SFrameW_PointDR],
							   variables[SFrameW_PointUR],
							   variables[SFrameW_Dist2]);
   constraints[SFrameW_ConstDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SFrameW_ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SFrameW_ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SFrameW_ConstDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SFrameW_ConstDRUR]->VarChoices(Scheme5, 1, 1, 1);
   constraints[SFrameW_ConstDRUR]->VarChoices(Scheme6, 0, 0, 0);
   constraints[SFrameW_ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SFrameW_ConstDRDL] = new DistanceConstraint("Const34",
							   NumSchemes,
							   variables[SFrameW_PointDR],
							   variables[SFrameW_PointDL],
							   variables[SFrameW_Dist1]);
   constraints[SFrameW_ConstDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SFrameW_ConstDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SFrameW_ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SFrameW_ConstDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SFrameW_ConstDRDL]->VarChoices(Scheme5, 1, 1, 1);
   constraints[SFrameW_ConstDRDL]->VarChoices(Scheme6, 0, 0, 0);
   constraints[SFrameW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[SFrameW_PointMatl] = PointWidgetMaterial;
   materials[SFrameW_EdgeMatl] = EdgeWidgetMaterial;
   materials[SFrameW_SliderMatl] = SliderWidgetMaterial;
   materials[SFrameW_HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = SFrameW_SphereUL, pick = SFrameW_PickSphUL;
	geom <= SFrameW_SphereDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[SFrameW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[SFrameW_PointMatl]);
   
   GeomGroup* resizes = new GeomGroup;
   for (geom = SFrameW_GeomResizeU, pick = SFrameW_PickResizeU;
	geom <= SFrameW_GeomResizeL; geom++, pick++) {
      geometries[geom] = new GeomCappedCylinder;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[SFrameW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      resizes->add(picks[pick]);
   }
   GeomMaterial* resizem = new GeomMaterial(resizes, materials[SFrameW_PointMatl]);

   GeomGroup* cyls = new GeomGroup;
   for (geom = SFrameW_CylU; geom <= SFrameW_CylL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[SFrameW_PickCyls] = new GeomPick(cyls, module);
   picks[SFrameW_PickCyls]->set_highlight(materials[SFrameW_HighMatl]);
   picks[SFrameW_PickCyls]->set_cbdata((void*)SFrameW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[SFrameW_PickCyls], materials[SFrameW_EdgeMatl]);

   GeomGroup* sliders = new GeomGroup;
   geometries[SFrameW_SliderCyl1] = new GeomCappedCylinder;
   picks[SFrameW_PickSlider1] = new GeomPick(geometries[SFrameW_SliderCyl1], module);
   picks[SFrameW_PickSlider1]->set_highlight(materials[SFrameW_HighMatl]);
   picks[SFrameW_PickSlider1]->set_cbdata((void*)SFrameW_PickSlider1);
   sliders->add(picks[SFrameW_PickSlider1]);
   geometries[SFrameW_SliderCyl2] = new GeomCappedCylinder;
   picks[SFrameW_PickSlider2] = new GeomPick(geometries[SFrameW_SliderCyl2], module);
   picks[SFrameW_PickSlider2]->set_highlight(materials[SFrameW_HighMatl]);
   picks[SFrameW_PickSlider2]->set_cbdata((void*)SFrameW_PickSlider2);
   sliders->add(picks[SFrameW_PickSlider2]);
   GeomMaterial* slidersm = new GeomMaterial(sliders, materials[SFrameW_SliderMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(resizem);
   w->add(cylsm);
   w->add(slidersm);
   
   SetEpsilon(widget_scale*1e-6);

   FinishWidget(w);
}


ScaledFrameWidget::~ScaledFrameWidget()
{
}


void
ScaledFrameWidget::widget_execute()
{
   ((GeomSphere*)geometries[SFrameW_SphereUL])->move(variables[SFrameW_PointUL]->GetPoint(),
						     1*widget_scale);
   ((GeomSphere*)geometries[SFrameW_SphereUR])->move(variables[SFrameW_PointUR]->GetPoint(),
						     1*widget_scale);
   ((GeomSphere*)geometries[SFrameW_SphereDR])->move(variables[SFrameW_PointDR]->GetPoint(),
						     1*widget_scale);
   ((GeomSphere*)geometries[SFrameW_SphereDL])->move(variables[SFrameW_PointDL]->GetPoint(),
						     1*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_CylU])->move(variables[SFrameW_PointUL]->GetPoint(),
						   variables[SFrameW_PointUR]->GetPoint(),
						   0.5*widget_scale);
   Point p(variables[SFrameW_PointUL]->GetPoint() + (variables[SFrameW_PointUR]->GetPoint()
						- variables[SFrameW_PointUL]->GetPoint()) / 2.0);
   ((GeomCappedCylinder*)geometries[SFrameW_GeomResizeU])->move(p - (GetAxis2() * 0.6 * widget_scale),
								p + (GetAxis2() * 0.6 * widget_scale),
								0.75*widget_scale);
   p = variables[SFrameW_PointUR]->GetPoint() + (variables[SFrameW_PointDR]->GetPoint()
					    - variables[SFrameW_PointUR]->GetPoint()) / 2.0;
   ((GeomCappedCylinder*)geometries[SFrameW_GeomResizeR])->move(p - (GetAxis1() * 0.6 * widget_scale),
								p + (GetAxis1() * 0.6 * widget_scale),
								0.75*widget_scale);
   p = variables[SFrameW_PointDR]->GetPoint() + (variables[SFrameW_PointDL]->GetPoint()
					    - variables[SFrameW_PointDR]->GetPoint()) / 2.0;
   ((GeomCappedCylinder*)geometries[SFrameW_GeomResizeD])->move(p - (GetAxis2() * 0.6 * widget_scale),
								p + (GetAxis2() * 0.6 * widget_scale),
								0.75*widget_scale);
   p = variables[SFrameW_PointDL]->GetPoint() + (variables[SFrameW_PointUL]->GetPoint()
					    - variables[SFrameW_PointDL]->GetPoint()) / 2.0;
   ((GeomCappedCylinder*)geometries[SFrameW_GeomResizeL])->move(p - (GetAxis1() * 0.6 * widget_scale),
								p + (GetAxis1() * 0.6 * widget_scale),
								0.75*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_CylR])->move(variables[SFrameW_PointUR]->GetPoint(),
						   variables[SFrameW_PointDR]->GetPoint(),
						   0.5*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_CylD])->move(variables[SFrameW_PointDR]->GetPoint(),
						   variables[SFrameW_PointDL]->GetPoint(),
						   0.5*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_CylL])->move(variables[SFrameW_PointDL]->GetPoint(),
						   variables[SFrameW_PointUL]->GetPoint(),
						   0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[SFrameW_SliderCyl1])->move(variables[SFrameW_Slider1]->GetPoint()
							       - (GetAxis1() * 0.3 * widget_scale),
							       variables[SFrameW_Slider1]->GetPoint()
							       + (GetAxis1() * 0.3 * widget_scale),
							       1.1*widget_scale);
   ((GeomCappedCylinder*)geometries[SFrameW_SliderCyl2])->move(variables[SFrameW_Slider2]->GetPoint()
							       - (GetAxis2() * 0.3 * widget_scale),
							       variables[SFrameW_Slider2]->GetPoint()
							       + (GetAxis2() * 0.3 * widget_scale),
							       1.1*widget_scale);

   SetEpsilon(widget_scale*1e-6);

   Vector spvec1(variables[SFrameW_PointUR]->GetPoint() - variables[SFrameW_PointUL]->GetPoint());
   Vector spvec2(variables[SFrameW_PointDL]->GetPoint() - variables[SFrameW_PointUL]->GetPoint());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if ((geom == SFrameW_PickResizeU) || (geom == SFrameW_PickResizeD) || (geom == SFrameW_PickSlider2))
	    picks[geom]->set_principal(spvec2);
	 else if ((geom == SFrameW_PickResizeL) || (geom == SFrameW_PickResizeR) || (geom == SFrameW_PickSlider1))
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
ScaledFrameWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			       void* cbdata )
{
   ((DistanceConstraint*)constraints[SFrameW_ConstSDist1])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[SFrameW_ConstSDist2])->SetDefault(GetAxis2());

   for (Index v=0; v<NumVars; v++)
      variables[v]->Reset();
   
   switch((int)cbdata){
   case SFrameW_PickSphUL:
      variables[SFrameW_PointUL]->SetDelta(delta);
      break;
   case SFrameW_PickSphUR:
      variables[SFrameW_PointUR]->SetDelta(delta);
      break;
   case SFrameW_PickSphDR:
      variables[SFrameW_PointDR]->SetDelta(delta);
      break;
   case SFrameW_PickSphDL:
      variables[SFrameW_PointDL]->SetDelta(delta);
      break;
   case SFrameW_PickResizeU:
      variables[SFrameW_PointUR]->SetDelta(delta, Scheme6);
      break;
   case SFrameW_PickResizeR:
      variables[SFrameW_PointDR]->SetDelta(delta, Scheme5);
      break;
   case SFrameW_PickResizeD:
      variables[SFrameW_PointDL]->SetDelta(delta, Scheme6);
      break;
   case SFrameW_PickResizeL:
      variables[SFrameW_PointUL]->SetDelta(delta, Scheme5);
      break;
   case SFrameW_PickSlider1:
      variables[SFrameW_Slider1]->SetDelta(delta);
      break;
   case SFrameW_PickSlider2:
      variables[SFrameW_Slider2]->SetDelta(delta);
      break;
   case SFrameW_PickCyls:
      variables[SFrameW_PointUL]->MoveDelta(delta);
      variables[SFrameW_PointUR]->MoveDelta(delta);
      variables[SFrameW_PointDR]->MoveDelta(delta);
      variables[SFrameW_PointDL]->MoveDelta(delta);
      variables[SFrameW_Slider1]->MoveDelta(delta);
      variables[SFrameW_Slider2]->MoveDelta(delta);
      break;
   }
}


void
ScaledFrameWidget::SetPosition( const Point& UL, const Point& UR, const Point& DL )
{
   Real size1((UR-UL).length()), size2((DL-UL).length());
   
   variables[SFrameW_PointUL]->Move(UL);
   variables[SFrameW_PointUR]->Move(UR);
   variables[SFrameW_PointDL]->Move(DL);
   variables[SFrameW_Dist1]->Move(size1);
   variables[SFrameW_Dist2]->Move(size2);
   variables[SFrameW_PointDR]->Set(UR+(DL-UL), Scheme5); // This should set Hypo...
   variables[SFrameW_SDist1]->Set(size1*variables[SFrameW_Ratio1]->GetReal(), Scheme1); // Slider1...
   variables[SFrameW_SDist2]->Set(size2*variables[SFrameW_Ratio2]->GetReal(), Scheme1); // Slider2...

   execute();
}


void
ScaledFrameWidget::GetPosition( Point& UL, Point& UR, Point& DL )
{
   UL = variables[SFrameW_PointUL]->GetPoint();
   UR = variables[SFrameW_PointUR]->GetPoint();
   DL = variables[SFrameW_PointDL]->GetPoint();
}


void
ScaledFrameWidget::SetPosition( const Point& center, const Vector& normal,
				const Real size1, const Real size2 )
{
   Real s1(size1/2.0), s2(size2/2.0);
   Vector axis1, axis2;
   normal.find_orthogonal(axis1, axis2);
   
   variables[SFrameW_PointUL]->Move(center-axis1*s1-axis2*s2);
   variables[SFrameW_PointDR]->Move(center+axis1*s1+axis2*s2);
   variables[SFrameW_PointUR]->Move(center+axis1*s1-axis2*s2);
   variables[SFrameW_PointDL]->Move(center-axis1*s1+axis2*s2);
   variables[SFrameW_Dist1]->Move(size1);
   variables[SFrameW_Dist2]->Set(size2); // This should set the Hypo...
   variables[SFrameW_SDist1]->Set(size1*variables[SFrameW_Ratio1]->GetReal(), Scheme1); // Slider1...
   variables[SFrameW_SDist2]->Set(size2*variables[SFrameW_Ratio2]->GetReal(), Scheme1); // Slider2...

   execute();
}


void
ScaledFrameWidget::GetPosition( Point& center, Vector& normal,
				Real& size1, Real& size2 )
{
   center = (variables[SFrameW_PointDR]->GetPoint()
	     + ((variables[SFrameW_PointUL]->GetPoint()-variables[SFrameW_PointDR]->GetPoint())
		/ 2.0));
   normal = Cross(GetAxis1(), GetAxis2());
   size1 = variables[SFrameW_Dist1]->GetReal();
   size2 = variables[SFrameW_Dist2]->GetReal();
}


void
ScaledFrameWidget::SetRatio1( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[SFrameW_Ratio1]->Set(ratio);
   
   execute();
}


Real
ScaledFrameWidget::GetRatio1() const
{
   return (variables[SFrameW_Ratio1]->GetReal());
}


void
ScaledFrameWidget::SetRatio2( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[SFrameW_Ratio2]->Set(ratio);
   
   execute();
}


Real
ScaledFrameWidget::GetRatio2() const
{
   return (variables[SFrameW_Ratio2]->GetReal());
}


void
ScaledFrameWidget::SetSize( const Real size1, const Real size2 )
{
   ASSERT((size1>=0.0)&&(size1>=0.0));

   Point center(variables[SFrameW_PointDR]->GetPoint()
		+ ((variables[SFrameW_PointUL]->GetPoint()-variables[SFrameW_PointDR]->GetPoint())
		   / 2.0));
   Vector axis1((variables[SFrameW_PointUR]->GetPoint() - variables[SFrameW_PointUL]->GetPoint())/2.0);
   Vector axis2((variables[SFrameW_PointDL]->GetPoint() - variables[SFrameW_PointUL]->GetPoint())/2.0);
   Real ratio1(size1/variables[SFrameW_Dist1]->GetReal());
   Real ratio2(size2/variables[SFrameW_Dist2]->GetReal());

   variables[SFrameW_PointUL]->Move(center-axis1*ratio1-axis2*ratio2);
   variables[SFrameW_PointDR]->Move(center+axis1*ratio1+axis2*ratio2);
   variables[SFrameW_PointUR]->Move(center+axis1*ratio1-axis2*ratio2);
   variables[SFrameW_PointDL]->Move(center-axis1*ratio1+axis2*ratio2);

   variables[SFrameW_Dist1]->Move(size1);
   variables[SFrameW_Dist2]->Set(size2); // This should set the Hypo...
   variables[SFrameW_SDist1]->Set(size1*variables[SFrameW_Ratio1]->GetReal(), Scheme1); // Slider1...
   variables[SFrameW_SDist2]->Set(size2*variables[SFrameW_Ratio2]->GetReal(), Scheme1); // Slider2...

   execute();
}

void
ScaledFrameWidget::GetSize( Real& size1, Real& size2 ) const
{
   size1 = variables[SFrameW_Dist1]->GetReal();
   size2 = variables[SFrameW_Dist2]->GetReal();
}

   
Vector
ScaledFrameWidget::GetAxis1()
{
   Vector axis(variables[SFrameW_PointUR]->GetPoint() - variables[SFrameW_PointUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


Vector
ScaledFrameWidget::GetAxis2()
{
   Vector axis(variables[SFrameW_PointDL]->GetPoint() - variables[SFrameW_PointUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


