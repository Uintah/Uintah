
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
const Index NumGeoms = 10;
const Index NumMatls = 3;
const Index NumPcks = 7;
const Index NumSchemes = 4;

enum { SFrameW_ConstULDR, SFrameW_ConstURDL, SFrameW_ConstPyth, SFrameW_ConstPlane,
       SFrameW_ConstULUR, SFrameW_ConstULDL, SFrameW_ConstDRUR, SFrameW_ConstDRDL,
       SFrameW_ConstLine1, SFrameW_ConstSDist1, SFrameW_ConstRatio1,
       SFrameW_ConstLine2, SFrameW_ConstSDist2, SFrameW_ConstRatio2 };
enum { SFrameW_SphereUL, SFrameW_SphereUR, SFrameW_SphereDR, SFrameW_SphereDL,
       SFrameW_CylU, SFrameW_CylR, SFrameW_CylD, SFrameW_CylL,
       SFrameW_SliderCyl1, SFrameW_SliderCyl2 };
enum { SFrameW_PointMatl, SFrameW_EdgeMatl, SFrameW_HighMatl };
enum { SFrameW_PickSphUL, SFrameW_PickSphUR, SFrameW_PickSphDR, SFrameW_PickSphDL, SFrameW_PickCyls,
       SFrameW_PickSlider1, SFrameW_PickSlider2 };

ScaledFrameWidget::ScaledFrameWidget( Module* module, Real widget_scale )
: BaseWidget(module, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1)
{
   cerr << "Starting ScaledFrameWidget CTOR" << endl;
   Real INIT = 1.0*widget_scale;
   variables[SFrameW_PointUL] = new Variable("PntUL", Scheme1, Point(0, 0, 0));
   variables[SFrameW_PointUR] = new Variable("PntUR", Scheme2, Point(INIT, 0, 0));
   variables[SFrameW_PointDR] = new Variable("PntDR", Scheme1, Point(INIT, INIT, 0));
   variables[SFrameW_PointDL] = new Variable("PntDL", Scheme2, Point(0, INIT, 0));
   variables[SFrameW_Slider1] = new Variable("Slider1", Scheme3, Point(INIT/2.0, 0, 0));
   variables[SFrameW_Slider2] = new Variable("Slider2", Scheme4, Point(0, INIT/2.0, 0));
   variables[SFrameW_Dist1] = new Variable("Dist1", Scheme1, Point(INIT, 0, 0));
   variables[SFrameW_Dist2] = new Variable("Dist2", Scheme1, Point(INIT, 0, 0));
   variables[SFrameW_Hypo] = new Variable("Hypo", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));
   variables[SFrameW_SDist1] = new Variable("SDist1", Scheme3, Point(INIT/2.0, 0, 0));
   variables[SFrameW_SDist2] = new Variable("SDist2", Scheme4, Point(INIT/2.0, 0, 0));
   variables[SFrameW_Ratio1] = new Variable("Ratio1", Scheme1, Point(0.5, 0, 0));
   variables[SFrameW_Ratio2] = new Variable("Ratio2", Scheme1, Point(0.5, 0, 0));

   constraints[SFrameW_ConstLine1] = new SegmentConstraint("ConstLine1",
							 NumSchemes,
							 variables[SFrameW_PointUL],
							 variables[SFrameW_PointUR],
							 variables[SFrameW_Slider1]);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme1, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme2, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[SFrameW_ConstLine1]->VarChoices(Scheme4, 2, 2, 2);
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
   constraints[SFrameW_ConstPlane]->Priorities(P_Highest, P_Highest,
					      P_Highest, P_Highest);
   constraints[SFrameW_ConstULDR] = new DistanceConstraint("Const13",
							  NumSchemes,
							  variables[SFrameW_PointUL],
							  variables[SFrameW_PointDR],
							  variables[SFrameW_Hypo]);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[SFrameW_ConstULDR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SFrameW_ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[SFrameW_ConstURDL] = new DistanceConstraint("Const24",
							  NumSchemes,
							  variables[SFrameW_PointUR],
							  variables[SFrameW_PointDL],
							  variables[SFrameW_Hypo]);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SFrameW_ConstURDL]->VarChoices(Scheme4, 2, 2, 1);
   constraints[SFrameW_ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[SFrameW_ConstPyth] = new PythagorasConstraint("ConstPyth",
							     NumSchemes,
							     variables[SFrameW_Dist1],
							     variables[SFrameW_Dist2],
							     variables[SFrameW_Hypo]);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme1, 2, 2, 0);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme3, 2, 2, 0);
   constraints[SFrameW_ConstPyth]->VarChoices(Scheme4, 2, 2, 1);
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
   constraints[SFrameW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[SFrameW_PointMatl] = new Material(Color(0,0,0), Color(.54, .60, 1),
					      Color(.5,.5,.5), 20);
   materials[SFrameW_EdgeMatl] = new Material(Color(0,0,0), Color(.54, .60, .66),
					     Color(.5,.5,.5), 20);
   materials[SFrameW_HighMatl] = new Material(Color(0,0,0), Color(.7,.7,.7),
					     Color(0,0,.6), 20);

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
   geometries[SFrameW_SliderCyl1] = new GeomCylinder;
   picks[SFrameW_PickSlider1] = new GeomPick(geometries[SFrameW_SliderCyl1], module);
   picks[SFrameW_PickSlider1]->set_highlight(materials[SFrameW_HighMatl]);
   picks[SFrameW_PickSlider1]->set_cbdata((void*)SFrameW_PickSlider1);
   sliders->add(picks[SFrameW_PickSlider1]);
   geometries[SFrameW_SliderCyl2] = new GeomCylinder;
   picks[SFrameW_PickSlider2] = new GeomPick(geometries[SFrameW_SliderCyl2], module);
   picks[SFrameW_PickSlider2]->set_highlight(materials[SFrameW_HighMatl]);
   picks[SFrameW_PickSlider2]->set_cbdata((void*)SFrameW_PickSlider2);
   sliders->add(picks[SFrameW_PickSlider2]);
   GeomMaterial* slidersm = new GeomMaterial(sliders, materials[SFrameW_PointMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(cylsm);
   w->add(slidersm);

   FinishWidget(w);
   
   SetEpsilon(widget_scale*1e-4);
   
   // Init variables.
   for (Index vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Order();
   
   for (vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Resolve();
   cerr << "Done with ScaledFrameWidget CTOR" << endl;
}


ScaledFrameWidget::~ScaledFrameWidget()
{
}


void
ScaledFrameWidget::execute()
{
   ((GeomSphere*)geometries[SFrameW_SphereUL])->move(variables[SFrameW_PointUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SFrameW_SphereUR])->move(variables[SFrameW_PointUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SFrameW_SphereDR])->move(variables[SFrameW_PointDR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SFrameW_SphereDL])->move(variables[SFrameW_PointDL]->Get(),
						    1*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_CylU])->move(variables[SFrameW_PointUL]->Get(),
						  variables[SFrameW_PointUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_CylR])->move(variables[SFrameW_PointUR]->Get(),
						  variables[SFrameW_PointDR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_CylD])->move(variables[SFrameW_PointDR]->Get(),
						  variables[SFrameW_PointDL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_CylL])->move(variables[SFrameW_PointDL]->Get(),
						  variables[SFrameW_PointUL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_SliderCyl1])->move(variables[SFrameW_Slider1]->Get()
							- (GetAxis1() * 0.2 * widget_scale),
							variables[SFrameW_Slider1]->Get()
							+ (GetAxis1() * 0.2 * widget_scale),
							1.0*widget_scale);
   ((GeomCylinder*)geometries[SFrameW_SliderCyl2])->move(variables[SFrameW_Slider2]->Get()
							- (GetAxis2() * 0.2 * widget_scale),
							variables[SFrameW_Slider2]->Get()
							+ (GetAxis2() * 0.2 * widget_scale),
							1.0*widget_scale);

   SetEpsilon(widget_scale*1e-4);

   Vector spvec1(variables[SFrameW_PointUR]->Get() - variables[SFrameW_PointUL]->Get());
   Vector spvec2(variables[SFrameW_PointDL]->Get() - variables[SFrameW_PointUL]->Get());
   spvec1.normalize();
   spvec2.normalize();
   Vector v = Cross(spvec1, spvec2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      if (geom == SFrameW_PickSlider1)
	 picks[geom]->set_principal(spvec1);
      else if (geom == SFrameW_PickSlider2)
	 picks[geom]->set_principal(spvec2);
      else
	 picks[geom]->set_principal(spvec1, spvec2, v);
   }
}

void
ScaledFrameWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   ((DistanceConstraint*)constraints[SFrameW_ConstSDist1])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[SFrameW_ConstSDist2])->SetDefault(GetAxis2());

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

