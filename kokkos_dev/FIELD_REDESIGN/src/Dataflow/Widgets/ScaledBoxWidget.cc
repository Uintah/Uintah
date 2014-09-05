//static char *id="@(#) $Id$";

/*
 *  ScaledBoxWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 *
 *
 *  Update Log
 *  ~~~~~~ ~~~
 *   7.1.95 -- DWM -- Added aligned member and methods (copied from  BoxWidget)
 */

#include <PSECore/Widgets/ScaledBoxWidget.h>
#include <PSECore/Constraints/DistanceConstraint.h>
#include <PSECore/Constraints/PythagorasConstraint.h>
#include <PSECore/Constraints/RatioConstraint.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Dataflow/Module.h>

namespace PSECore {
namespace Widgets {

using SCICore::GeomSpace::GeomGroup;
using SCICore::GeomSpace::GeomSphere;
using SCICore::GeomSpace::GeomCylinder;
using SCICore::GeomSpace::GeomCappedCylinder;

using namespace PSECore::Constraints;

const Index NumCons = 12;
const Index NumVars = 16;
const Index NumGeoms = 35;
const Index NumPcks = 16;
const Index NumMatls = 4;
const Index NumMdes = 7;
const Index NumSwtchs = 4;
const Index NumSchemes = 7;

enum { ConstRD, ConstDI, ConstIR, ConstRC, ConstDC, ConstIC,
       ConstPythRD, ConstPythDI, ConstPythIR,
       ConstRatioR, ConstRatioD, ConstRatioI };
enum { SphereR, SphereL, SphereD, SphereU, SphereI, SphereO,
       SmallSphereIUL, SmallSphereIUR, SmallSphereIDR, SmallSphereIDL,
       SmallSphereOUL, SmallSphereOUR, SmallSphereODR, SmallSphereODL,
       CylIU, CylIR, CylID, CylIL,
       CylMU, CylMR, CylMD, CylML,
       CylOU, CylOR, CylOD, CylOL,
       GeomResizeR, GeomResizeL, GeomResizeD, GeomResizeU,
       GeomResizeI, GeomResizeO,
       SliderCylR, SliderCylD, SliderCylI };
enum { PickSphR, PickSphL, PickSphD, PickSphU, PickSphI, PickSphO,
       PickCyls,
       PickResizeR, PickResizeL, PickResizeD, PickResizeU,
       PickResizeI, PickResizeO,
       PickSliderR, PickSliderD, PickSliderI };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
ScaledBoxWidget::ScaledBoxWidget( Module* module, CrowdMonitor* lock, 
				 double widget_scale, Index aligned )
: BaseWidget(module, lock, "ScaledBoxWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  oldrightaxis(1, 0, 0), olddownaxis(0, 1, 0), oldinaxis(0, 0, 1),
  aligned(aligned)
{
   Real INIT = 5.0*widget_scale;
   variables[CenterVar] = scinew PointVariable("Center", solve, Scheme1, Point(0, 0, 0));
   variables[PointRVar] = scinew PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
   variables[PointDVar] = scinew PointVariable("PntD", solve, Scheme2, Point(0, INIT, 0));
   variables[PointIVar] = scinew PointVariable("PntI", solve, Scheme3, Point(0, 0, INIT));
   variables[DistRVar] = scinew RealVariable("DISTR", solve, Scheme4, INIT);
   variables[DistDVar] = scinew RealVariable("DISTD", solve, Scheme5, INIT);
   variables[DistIVar] = scinew RealVariable("DISTI", solve, Scheme6, INIT);
   variables[HypoRDVar] = scinew RealVariable("HYPOR", solve, Scheme4, sqrt(2*INIT*INIT));
   variables[HypoDIVar] = scinew RealVariable("HYPOD", solve, Scheme5, sqrt(2*INIT*INIT));
   variables[HypoIRVar] = scinew RealVariable("HYPOI", solve, Scheme6, sqrt(2*INIT*INIT));
   variables[SDistRVar] = scinew RealVariable("SDistR", solve, Scheme7, INIT/2.0);
   variables[SDistDVar] = scinew RealVariable("SDistD", solve, Scheme7, INIT/2.0);
   variables[SDistIVar] = scinew RealVariable("SDistI", solve, Scheme7, INIT/2.0);
   variables[RatioRVar] = scinew RealVariable("RatioR", solve, Scheme1, 0.5);
   variables[RatioDVar] = scinew RealVariable("RatioD", solve, Scheme1, 0.5);
   variables[RatioIVar] = scinew RealVariable("RatioI", solve, Scheme1, 0.5);

   constraints[ConstRatioR] = scinew RatioConstraint("ConstRatioR",
						  NumSchemes,
						  variables[SDistRVar],
						  variables[DistRVar],
						  variables[RatioRVar]);
   constraints[ConstRatioR]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme7, 2, 2, 2);
   constraints[ConstRatioR]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRatioD] = scinew RatioConstraint("ConstRatioD",
						  NumSchemes,
						  variables[SDistDVar],
						  variables[DistDVar],
						  variables[RatioDVar]);
   constraints[ConstRatioD]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme7, 2, 2, 2);
   constraints[ConstRatioD]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRatioI] = scinew RatioConstraint("ConstRatioI",
						  NumSchemes,
						  variables[SDistIVar],
						  variables[DistIVar],
						  variables[RatioIVar]);
   constraints[ConstRatioI]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatioI]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRatioI]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRatioI]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstRatioI]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ConstRatioI]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstRatioI]->VarChoices(Scheme7, 2, 2, 2);
   constraints[ConstRatioI]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRD] = scinew DistanceConstraint("ConstRD",
						 NumSchemes,
						 variables[PointRVar],
						 variables[PointDVar],
						 variables[HypoRDVar]);
   constraints[ConstRD]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstRD]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRD]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRD]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstRD]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstRD]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstRD]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstRD]->Priorities(P_Default, P_Default, P_Default);
   constraints[ConstPythRD] = scinew PythagorasConstraint("ConstPythRD",
						     NumSchemes,
						     variables[DistRVar],
						     variables[DistDVar],
						     variables[HypoRDVar]);
   constraints[ConstPythRD]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPythRD]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstPythRD]->VarChoices(Scheme3, 1, 0, 0);
   constraints[ConstPythRD]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstPythRD]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstPythRD]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstPythRD]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstPythRD]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstDI] = scinew DistanceConstraint("ConstDI",
						 NumSchemes,
						 variables[PointDVar],
						 variables[PointIVar],
						 variables[HypoDIVar]);
   constraints[ConstDI]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstDI]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstDI]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstDI]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstDI]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstDI]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstDI]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstDI]->Priorities(P_Default, P_Default, P_Default);
   constraints[ConstPythDI] = scinew PythagorasConstraint("ConstPythDI",
						     NumSchemes,
						     variables[DistDVar],
						     variables[DistIVar],
						     variables[HypoDIVar]);
   constraints[ConstPythDI]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPythDI]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstPythDI]->VarChoices(Scheme3, 1, 0, 0);
   constraints[ConstPythDI]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstPythDI]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstPythDI]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstPythDI]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstPythDI]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstIR] = scinew DistanceConstraint("ConstIR",
						 NumSchemes,
						 variables[PointIVar],
						 variables[PointRVar],
						 variables[HypoIRVar]);
   constraints[ConstIR]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstIR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstIR]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstIR]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstIR]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstIR]->Priorities(P_Default, P_Default, P_Default);
   constraints[ConstPythIR] = scinew PythagorasConstraint("ConstPythIR",
						     NumSchemes,
						     variables[DistIVar],
						     variables[DistRVar],
						     variables[HypoIRVar]);
   constraints[ConstPythIR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPythIR]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstPythIR]->VarChoices(Scheme3, 1, 0, 0);
   constraints[ConstPythIR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstPythIR]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstPythIR]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstPythIR]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstPythIR]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRC] = scinew DistanceConstraint("ConstRC",
						 NumSchemes,
						 variables[PointRVar],
						 variables[CenterVar],
						 variables[DistRVar]);
   constraints[ConstRC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstRC]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstRC]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstDC] = scinew DistanceConstraint("ConstDC",
					       NumSchemes,
					       variables[PointDVar],
					       variables[CenterVar],
					       variables[DistDVar]);
   constraints[ConstDC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme5, 2, 2, 2);
   constraints[ConstDC]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstDC]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstIC] = scinew DistanceConstraint("ConstIC",
					       NumSchemes,
					       variables[PointIVar],
					       variables[CenterVar],
					       variables[DistIVar]);
   constraints[ConstIC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme6, 2, 2, 2);
   constraints[ConstIC]->VarChoices(Scheme7, 1, 0, 1);
   constraints[ConstIC]->Priorities(P_Highest, P_Highest, P_Default);

   Index geom, pick;
   GeomGroup* cyls = scinew GeomGroup;
   for (geom = SmallSphereIUL; geom <= SmallSphereODL; geom++) {
      geometries[geom] = scinew GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = CylIU; geom <= CylOL; geom++) {
      geometries[geom] = scinew GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[PickCyls] = scinew GeomPick(cyls, module, this, PickCyls);
   picks[PickCyls]->set_highlight(DefaultHighlightMaterial);
   materials[EdgeMatl] = scinew GeomMaterial(picks[PickCyls], DefaultEdgeMaterial);
   CreateModeSwitch(0, materials[EdgeMatl]);

   GeomGroup* pts = scinew GeomGroup;
   for (geom = SphereR, pick = PickSphR;
	geom <= SphereO; geom++, pick++) {
      geometries[geom] = scinew GeomSphere;
      picks[pick] = scinew GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(DefaultHighlightMaterial);
      pts->add(picks[pick]);
   }
   materials[PointMatl] = scinew GeomMaterial(pts, DefaultPointMaterial);
   CreateModeSwitch(1, materials[PointMatl]);
   
   GeomGroup* resizes = scinew GeomGroup;
   for (geom = GeomResizeR, pick = PickResizeR;
	geom <= GeomResizeO; geom++, pick++) {
      geometries[geom] = scinew GeomCappedCylinder;
      picks[pick] = scinew GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(DefaultHighlightMaterial);
      resizes->add(picks[pick]);
   }
   materials[ResizeMatl] = scinew GeomMaterial(resizes, DefaultResizeMaterial);
   CreateModeSwitch(2, materials[ResizeMatl]);

   GeomGroup* sliders = scinew GeomGroup;
   geometries[SliderCylR] = scinew GeomCappedCylinder;
   picks[PickSliderR] = scinew GeomPick(geometries[SliderCylR], module, this, PickSliderR);
   picks[PickSliderR]->set_highlight(DefaultHighlightMaterial);
   sliders->add(picks[PickSliderR]);
   geometries[SliderCylD] = scinew GeomCappedCylinder;
   picks[PickSliderD] = scinew GeomPick(geometries[SliderCylD], module, this, PickSliderD);
   picks[PickSliderD]->set_highlight(DefaultHighlightMaterial);
   sliders->add(picks[PickSliderD]);
   geometries[SliderCylI] = scinew GeomCappedCylinder;
   picks[PickSliderI] = scinew GeomPick(geometries[SliderCylI], module, this, PickSliderI);
   picks[PickSliderI]->set_highlight(DefaultHighlightMaterial);
   sliders->add(picks[PickSliderI]);
   materials[SliderMatl] = scinew GeomMaterial(sliders, DefaultSliderMaterial);
   CreateModeSwitch(3, materials[SliderMatl]);

   // Switch0 are the bars
   // Switch1 are the rotation points
   // Switch2 are the resize cylinders
   // Switch3 are the sliders
   SetMode(Mode0, Switch0|Switch1|Switch2|Switch3);
   SetMode(Mode1, Switch0|Switch1|Switch2);
   SetMode(Mode2, Switch0|Switch1);
   SetMode(Mode3, Switch0);
   SetMode(Mode4, Switch0|Switch1|Switch3);
   SetMode(Mode5, Switch0|Switch2|Switch3);
   SetMode(Mode6, Switch0|Switch2);

   FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
ScaledBoxWidget::~ScaledBoxWidget()
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
ScaledBoxWidget::redraw()
{
   Real sphererad(widget_scale), resizerad(0.5*widget_scale), cylinderrad(0.5*widget_scale);
   Vector Right(GetRightAxis()*variables[DistRVar]->real());
   Vector Down(GetDownAxis()*variables[DistDVar]->real());
   Vector In(GetInAxis()*variables[DistIVar]->real());
   Point Center(variables[CenterVar]->point());
   Point IUL(Center-Right-Down+In);
   Point IUR(Center+Right-Down+In);
   Point IDR(Center+Right+Down+In);
   Point IDL(Center-Right+Down+In);
   Point OUL(Center-Right-Down-In);
   Point OUR(Center+Right-Down-In);
   Point ODR(Center+Right+Down-In);
   Point ODL(Center-Right+Down-In);
   Point U(Center-Down);
   Point R(Center+Right);
   Point D(Center+Down);
   Point L(Center-Right);
   Point I(Center+In);
   Point O(Center-In);
   
   // draw the edges
   if (mode_switches[0]->get_state()) {
      ((GeomCylinder*)geometries[CylIU])->move(IUL, IUR, cylinderrad);
      ((GeomCylinder*)geometries[CylIR])->move(IUR, IDR, cylinderrad);
      ((GeomCylinder*)geometries[CylID])->move(IDR, IDL, cylinderrad);
      ((GeomCylinder*)geometries[CylIL])->move(IDL, IUL, cylinderrad);
      ((GeomCylinder*)geometries[CylMU])->move(IUL, OUL, cylinderrad);
      ((GeomCylinder*)geometries[CylMR])->move(IUR, OUR, cylinderrad);
      ((GeomCylinder*)geometries[CylMD])->move(IDR, ODR, cylinderrad);
      ((GeomCylinder*)geometries[CylML])->move(IDL, ODL, cylinderrad);
      ((GeomCylinder*)geometries[CylOU])->move(OUL, OUR, cylinderrad);
      ((GeomCylinder*)geometries[CylOR])->move(OUR, ODR, cylinderrad);
      ((GeomCylinder*)geometries[CylOD])->move(ODR, ODL, cylinderrad);
      ((GeomCylinder*)geometries[CylOL])->move(ODL, OUL, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereIUL])->move(IUL, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereIUR])->move(IUR, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereIDR])->move(IDR, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereIDL])->move(IDL, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereOUL])->move(OUL, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereOUR])->move(OUR, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereODR])->move(ODR, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereODL])->move(ODL, cylinderrad);
   }

   // draw the rotating points
   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[SphereR])->move(R, sphererad);
      ((GeomSphere*)geometries[SphereL])->move(L, sphererad);
      ((GeomSphere*)geometries[SphereD])->move(D, sphererad);
      ((GeomSphere*)geometries[SphereU])->move(U, sphererad);
      ((GeomSphere*)geometries[SphereI])->move(I, sphererad);
      ((GeomSphere*)geometries[SphereO])->move(O, sphererad);
   }

   // draw the resizing cylinders
   if (mode_switches[2]->get_state()) {
      Vector resizeR(GetRightAxis()*1.5*widget_scale),
	 resizeD(GetDownAxis()*1.5*widget_scale),
	 resizeI(GetInAxis()*1.5*widget_scale);
      
      ((GeomCappedCylinder*)geometries[GeomResizeR])->move(R, R + resizeR, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeL])->move(L, L - resizeR, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeD])->move(D, D + resizeD, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeU])->move(U, U - resizeD, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeI])->move(I, I + resizeI, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeO])->move(O, O - resizeI, resizerad);
   }

   // draw the sliders
   if (mode_switches[3]->get_state()) {
      Point SliderR(OUL+GetRightAxis()*variables[SDistRVar]->real()*2.0);
      Point SliderD(OUL+GetDownAxis()*variables[SDistDVar]->real()*2.0);
      Point SliderI(OUL+GetInAxis()*variables[SDistIVar]->real()*2.0);
      ((GeomCappedCylinder*)geometries[SliderCylR])->move(SliderR - (GetRightAxis() * 0.3 * widget_scale),
							  SliderR + (GetRightAxis() * 0.3 * widget_scale),
							  1.1*widget_scale);
      ((GeomCappedCylinder*)geometries[SliderCylD])->move(SliderD - (GetDownAxis() * 0.3 * widget_scale),
							  SliderD + (GetDownAxis() * 0.3 * widget_scale),
							  1.1*widget_scale);
      ((GeomCappedCylinder*)geometries[SliderCylI])->move(SliderI - (GetInAxis() * 0.3 * widget_scale),
							  SliderI + (GetInAxis() * 0.3 * widget_scale),
							  1.1*widget_scale);
   }

   Right.normalize();
   Down.normalize();
   In.normalize();
   for (Index geom = 0; geom < NumPcks; geom++) {
      if ((geom == PickResizeU) || (geom == PickResizeD) || (geom == PickSliderD))
	 picks[geom]->set_principal(Down);
      else if ((geom == PickResizeL) || (geom == PickResizeR) || (geom == PickSliderR))
	 picks[geom]->set_principal(Right);
      else if ((geom == PickResizeO) || (geom == PickResizeI) || (geom == PickSliderI))
	 picks[geom]->set_principal(In);
      else if ((geom == PickSphL) || (geom == PickSphR))
	 picks[geom]->set_principal(Down, In);
      else if ((geom == PickSphU) || (geom == PickSphD))
	 picks[geom]->set_principal(Right, In);
      else if ((geom == PickSphO) || (geom == PickSphI))
	 picks[geom]->set_principal(Right, Down);
      else
	 picks[geom]->set_principal(Right, Down, In);
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
ScaledBoxWidget::geom_moved( GeomPick*, int axis, double dist,
			     const Vector& delta, int pick, const BState& )
{
   switch(pick){
   case PickSphU:
       if (!aligned)
      variables[PointDVar]->SetDelta(-delta);
      break;
   case PickSphR:
       if (!aligned)
      variables[PointRVar]->SetDelta(delta);
      break;
   case PickSphD:
       if (!aligned)
      variables[PointDVar]->SetDelta(delta);
      break;
   case PickSphL:
       if (!aligned)
      variables[PointRVar]->SetDelta(-delta);
      break;
   case PickSphI:
       if (!aligned)
      variables[PointIVar]->SetDelta(delta);
      break;
   case PickSphO:
       if (!aligned)
      variables[PointIVar]->SetDelta(-delta);
      break;
   case PickResizeU:
      variables[PointRVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->MoveDelta(delta/2.0);
      variables[CenterVar]->SetDelta(delta/2.0, Scheme5);
      break;
   case PickResizeR:
      variables[CenterVar]->MoveDelta(delta/2.0);
      variables[PointDVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->MoveDelta(delta/2.0);
      variables[PointRVar]->SetDelta(delta, Scheme4);
      break;
   case PickResizeD:
      variables[CenterVar]->MoveDelta(delta/2.0);
      variables[PointRVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->MoveDelta(delta/2.0);
      variables[PointDVar]->SetDelta(delta, Scheme5);
      break;
   case PickResizeL:
      variables[PointDVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->MoveDelta(delta/2.0);
      variables[CenterVar]->SetDelta(delta/2.0, Scheme4);
      break;
   case PickResizeI:
      variables[CenterVar]->MoveDelta(delta/2.0);
      variables[PointRVar]->MoveDelta(delta/2.0);
      variables[PointDVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->SetDelta(delta, Scheme6);
      break;
   case PickResizeO:
      variables[PointRVar]->MoveDelta(delta/2.0);
      variables[PointDVar]->MoveDelta(delta/2.0);
      variables[CenterVar]->SetDelta(delta/2.0, Scheme6);
      break;
   case PickSliderR:
      {
	  if (axis==1) dist*=-1.0;
	  Real sdist(variables[SDistRVar]->real()+dist/2.0);
	  if (sdist<0.0) sdist=0.0;
	  else if (sdist>variables[DistRVar]->real()) sdist=variables[DistRVar]->real();
	  variables[SDistRVar]->Set(sdist);
      }
      break;
   case PickSliderD:
      {
	  if (axis==1) dist*=-1.0;
	  Real sdist = variables[SDistDVar]->real()+dist/2.0;
	  if (sdist<0.0) sdist=0.0;
	  else if (sdist>variables[DistDVar]->real()) sdist=variables[DistDVar]->real();
	  variables[SDistDVar]->Set(sdist);
      }
      break;
   case PickSliderI:
      {
	  if (axis==1) dist*=-1.0;
	  Real sdist = variables[SDistIVar]->real()+dist/2.0;
	  if (sdist<0.0) sdist=0.0;
	  else if (sdist>variables[DistIVar]->real()) sdist=variables[DistIVar]->real();
	  variables[SDistIVar]->Set(sdist);
      }
      break;
   case PickCyls:
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
ScaledBoxWidget::MoveDelta( const Vector& delta )
{
   variables[CenterVar]->MoveDelta(delta);
   variables[PointRVar]->MoveDelta(delta);
   variables[PointDVar]->MoveDelta(delta);
   variables[PointIVar]->MoveDelta(delta);

   execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
ScaledBoxWidget::ReferencePoint() const
{
   return variables[CenterVar]->point();
}


void
ScaledBoxWidget::SetPosition( const Point& center, const Point& R, const Point& D,
			      const Point& I )
{
   variables[PointRVar]->Move(R);
   variables[PointDVar]->Move(D);
   variables[PointIVar]->Move(I);
   Real sizeR((R-center).length());
   Real sizeD((D-center).length());
   Real sizeI((I-center).length());
   variables[DistRVar]->Move(sizeR);
   variables[DistDVar]->Move(sizeD);
   variables[DistIVar]->Move(sizeI);
   variables[CenterVar]->Set(center, Scheme3); // This should set Hypo...
   variables[SDistRVar]->Set(sizeR*variables[RatioRVar]->real(), Scheme1); // Slider1...
   variables[SDistDVar]->Set(sizeD*variables[RatioDVar]->real(), Scheme1); // Slider2...
   variables[SDistIVar]->Set(sizeI*variables[RatioIVar]->real(), Scheme1); // Slider3...

   execute(0);
}


void
ScaledBoxWidget::GetPosition( Point& center, Point& R, Point& D, Point& I )
{
   center = variables[CenterVar]->point();
   R = variables[PointRVar]->point();
   D = variables[PointDVar]->point();
   I = variables[PointIVar]->point();
}


void
ScaledBoxWidget::SetRatioR( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[RatioRVar]->Set(ratio);
   
   execute(0);
}


Real
ScaledBoxWidget::GetRatioR() const
{
   return (variables[RatioRVar]->real());
}


void
ScaledBoxWidget::SetRatioD( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[RatioDVar]->Set(ratio);
   
   execute(0);
}


Real
ScaledBoxWidget::GetRatioD() const
{
   return (variables[RatioDVar]->real());
}


void
ScaledBoxWidget::SetRatioI( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[RatioIVar]->Set(ratio);
   
   execute(0);
}


Real
ScaledBoxWidget::GetRatioI() const
{
   return (variables[RatioIVar]->real());
}


const Vector&
ScaledBoxWidget::GetRightAxis()
{
   Vector axis(variables[PointRVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldrightaxis;
   else
      return (oldrightaxis = axis.normal());
}


const Vector&
ScaledBoxWidget::GetDownAxis()
{
   Vector axis(variables[PointDVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return olddownaxis;
   else
      return (olddownaxis = axis.normal());
}


const Vector&
ScaledBoxWidget::GetInAxis()
{
   Vector axis(variables[PointIVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldinaxis;
   else
      return (oldinaxis = axis.normal());
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
clString
ScaledBoxWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   
   switch(mindex){
   case 0:
      return "Point";
   case 1:
      return "Edge";
   case 2:
      return "Resize";
   case 3:
      return "Slider";
   default:
      return "UnknownMaterial";
   }
}



Index
ScaledBoxWidget::IsAxisAligned() const
{
   return aligned;
}


void
ScaledBoxWidget::AxisAligned( const Index yesno )
{
   if (aligned == yesno) return;
   
   aligned = yesno;

   if (aligned) {
      Point center(variables[CenterVar]->point());
      // Shouldn't need to resolve constraints...
      variables[PointRVar]->Move(center+Vector(1,0,0)*variables[DistRVar]->real());
      variables[PointDVar]->Move(center+Vector(0,1,0)*variables[DistDVar]->real());
      variables[PointIVar]->Move(center+Vector(0,0,1)*variables[DistIVar]->real());
      oldrightaxis = Vector(1,0,0);
      olddownaxis = Vector(0,1,0);
      oldinaxis = Vector(0,0,1);
   }
   
   execute(0);
}

} // End namespace Widgets
} // End namespace PSECore

//
// $Log$
// Revision 1.2.2.3  2000/10/26 14:16:57  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.4  2000/06/26 21:54:18  bigler
// Added new modes to allow for more control over which pieces
// are rendered.
//
// Revision 1.3  2000/06/21 20:57:25  bigler
// Added additional modes for widget grid.
// One of the additional modes create a scaled widget frame that allows for resizing, but restricts movement to be axis alligned (no rotation).
// The other mode allows for rotation without resizing.
//
// Revision 1.2  1999/08/17 06:38:32  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:08  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//
