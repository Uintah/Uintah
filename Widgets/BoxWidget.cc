
/*
 *  BoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/BoxWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 18;
const Index NumVars = 11;
const Index NumGeoms = 52;
const Index NumPcks = 15;
const Index NumMdes = 4;
const Index NumSwtchs = 3;
const Index NumSchemes = 4;

enum { ConstIULODR, ConstOULIDR, ConstIDLOUR, ConstODLIUR,
       ConstHypo, ConstDiag,
       ConstIULUR, ConstIULDL, ConstIDRUR, ConstIDRDL,
       ConstMULUL, ConstMURUR, ConstMDRDR, ConstMDLDL,
       ConstOULUR, ConstOULDL, ConstODRUR, ConstODRDL };
enum { SphereIUL, SphereIUR, SphereIDR, SphereIDL,
       SphereOUL, SphereOUR, SphereODR, SphereODL,
       SmallSphereIUL, SmallSphereIUR, SmallSphereIDR, SmallSphereIDL,
       SmallSphereOUL, SmallSphereOUR, SmallSphereODR, SmallSphereODL,
       CylIU, CylIR, CylID, CylIL,
       CylMU, CylMR, CylMD, CylML,
       CylOU, CylOR, CylOD, CylOL,
       GeomResizeUU, GeomResizeUR, GeomResizeUD, GeomResizeUL,
       GeomResizeRU, GeomResizeRR, GeomResizeRD, GeomResizeRL,
       GeomResizeDU, GeomResizeDR, GeomResizeDD, GeomResizeDL,
       GeomResizeLU, GeomResizeLR, GeomResizeLD, GeomResizeLL,
       GeomResizeIU, GeomResizeIR, GeomResizeID, GeomResizeIL,
       GeomResizeOU, GeomResizeOR, GeomResizeOD, GeomResizeOL };
enum { PickSphIUL, PickSphIUR, PickSphIDR, PickSphIDL,
       PickSphOUL, PickSphOUR, PickSphODR, PickSphODL,
       PickCyls, PickResizeU, PickResizeR, PickResizeD,
       PickResizeL, PickResizeI, PickResizeO };

BoxWidget::BoxWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumPcks, NumMdes, NumSwtchs, widget_scale)
{
   Real INIT = 1.0*widget_scale;
   variables[PointIULVar] = new PointVariable("PntIUL", solve, Scheme1, Point(0, 0, 0));
   variables[PointIURVar] = new PointVariable("PntIUR", solve, Scheme2, Point(INIT, 0, 0));
   variables[PointIDRVar] = new PointVariable("PntIDR", solve, Scheme1, Point(INIT, INIT, 0));
   variables[PointIDLVar] = new PointVariable("PntIDL", solve, Scheme2, Point(0, INIT, 0));
   variables[PointOULVar] = new PointVariable("PntOUL", solve, Scheme1, Point(0, 0, INIT));
   variables[PointOURVar] = new PointVariable("PntOUR", solve, Scheme2, Point(INIT, 0, INIT));
   variables[PointODRVar] = new PointVariable("PntODR", solve, Scheme1, Point(INIT, INIT, INIT));
   variables[PointODLVar] = new PointVariable("PntODL", solve, Scheme2, Point(0, INIT, INIT));
   variables[DistVar] = new RealVariable("DIST", solve, Scheme1, INIT);
   variables[HypoVar] = new RealVariable("HYPO", solve, Scheme1, sqrt(2*INIT*INIT));
   variables[DiagVar] = new RealVariable("DIAG", solve, Scheme1, sqrt(3*INIT*INIT));

   NOT_FINISHED("Constraints not right!");
   
   constraints[ConstIULODR] = new DistanceConstraint("ConstIULODR",
						     NumSchemes,
						     variables[PointIULVar],
						     variables[PointODRVar],
						     variables[DiagVar]);
   constraints[ConstIULODR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[ConstIULODR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstIULODR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[ConstIULODR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstIULODR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstOULIDR] = new DistanceConstraint("ConstOULIDR",
						     NumSchemes,
						     variables[PointOULVar],
						     variables[PointIDRVar],
						     variables[DiagVar]);
   constraints[ConstOULIDR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstOULIDR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[ConstOULIDR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstOULIDR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstOULIDR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstIDLOUR] = new DistanceConstraint("ConstIDLOUR",
						     NumSchemes,
						     variables[PointIDLVar],
						     variables[PointOURVar],
						     variables[DiagVar]);
   constraints[ConstIDLOUR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[ConstIDLOUR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstIDLOUR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[ConstIDLOUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstIDLOUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstODLIUR] = new DistanceConstraint("ConstODLIUR",
						     NumSchemes,
						     variables[PointODLVar],
						     variables[PointIURVar],
						     variables[DiagVar]);
   constraints[ConstODLIUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstODLIUR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[ConstODLIUR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstODLIUR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstODLIUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstHypo] = new PythagorasConstraint("ConstHypo",
						     NumSchemes,
						     variables[DistVar],
						     variables[DistVar],
						     variables[HypoVar]);
   constraints[ConstHypo]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstHypo]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstDiag] = new PythagorasConstraint("ConstDiag",
						     NumSchemes,
						     variables[DistVar],
						     variables[HypoVar],
						     variables[DiagVar]);
   constraints[ConstDiag]->VarChoices(Scheme1, 2, 2, 1);
   constraints[ConstDiag]->VarChoices(Scheme2, 2, 2, 1);
   constraints[ConstDiag]->VarChoices(Scheme3, 2, 2, 1);
   constraints[ConstDiag]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstDiag]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstIULUR] = new DistanceConstraint("ConstIULUR",
						    NumSchemes,
						    variables[PointIULVar],
						    variables[PointIURVar],
						    variables[DistVar]);
   constraints[ConstIULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstIULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstIULDL] = new DistanceConstraint("ConstIULDL",
						    NumSchemes,
						    variables[PointIULVar],
						    variables[PointIDLVar],
						    variables[DistVar]);
   constraints[ConstIULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstIULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstIDRUR] = new DistanceConstraint("ConstIDRUR",
						    NumSchemes,
						    variables[PointIDRVar],
						    variables[PointIURVar],
						    variables[DistVar]);
   constraints[ConstIDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstIDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstIDRDL] = new DistanceConstraint("ConstIDRUR",
						    NumSchemes,
						    variables[PointIDRVar],
						    variables[PointIDLVar],
						    variables[DistVar]);
   constraints[ConstIDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstIDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIDRDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstMULUL] = new DistanceConstraint("ConstMULUL",
						    NumSchemes,
						    variables[PointIULVar],
						    variables[PointOULVar],
						    variables[DistVar]);
   constraints[ConstMULUL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstMULUL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstMULUL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstMULUL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstMULUL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstMURUR] = new DistanceConstraint("ConstMURUR",
						    NumSchemes,
						    variables[PointIURVar],
						    variables[PointOURVar],
						    variables[DistVar]);
   constraints[ConstMURUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstMURUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstMURUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstMURUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstMURUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstMDRDR] = new DistanceConstraint("ConstMDRDR",
						    NumSchemes,
						    variables[PointIDRVar],
						    variables[PointODRVar],
						    variables[DistVar]);
   constraints[ConstMDRDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstMDRDR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstMDRDR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstMDRDR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstMDRDR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstMDLDL] = new DistanceConstraint("ConstMDLDL",
						    NumSchemes,
						    variables[PointIDLVar],
						    variables[PointODLVar],
						    variables[DistVar]);
   constraints[ConstMDLDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstMDLDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstMDLDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstMDLDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstMDLDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstOULUR] = new DistanceConstraint("ConstOULUR",
						    NumSchemes,
						    variables[PointOULVar],
						    variables[PointOURVar],
						    variables[DistVar]);
   constraints[ConstOULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstOULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstOULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstOULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstOULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstOULDL] = new DistanceConstraint("ConstOULDL",
						    NumSchemes,
						    variables[PointOULVar],
						    variables[PointODLVar],
						    variables[DistVar]);
   constraints[ConstOULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstOULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstOULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstOULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstOULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstODRUR] = new DistanceConstraint("ConstODRUR",
						    NumSchemes,
						    variables[PointODRVar],
						    variables[PointOURVar],
						    variables[DistVar]);
   constraints[ConstODRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstODRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstODRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstODRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstODRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstODRDL] = new DistanceConstraint("ConstODRDL",
						    NumSchemes,
						    variables[PointODRVar],
						    variables[PointODLVar],
						    variables[DistVar]);
   constraints[ConstODRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstODRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstODRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstODRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstODRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   Index geom, pick;
   GeomGroup* cyls = new GeomGroup;
   for (geom = SmallSphereIUL; geom <= SmallSphereODL; geom++) {
      geometries[geom] = new GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = CylIU; geom <= CylOL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[PickCyls] = new GeomPick(cyls, module, this, PickCyls);
   picks[PickCyls]->set_highlight(HighlightMaterial);
   GeomMaterial* cylsm = new GeomMaterial(picks[PickCyls], EdgeMaterial);
   CreateModeSwitch(0, cylsm);

   GeomGroup* pts = new GeomGroup;
   for (geom = SphereIUL, pick = PickSphIUL;
	geom <= SphereODL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(HighlightMaterial);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, PointMaterial);
   CreateModeSwitch(1, ptsm);
   
   GeomGroup* resizes = new GeomGroup;
   GeomGroup* face;
   for (geom = GeomResizeUU, pick = PickResizeU;
	geom <= GeomResizeOL; geom+=4, pick++) {
      face = new GeomGroup;
      for (Index geom2=geom; geom2<geom+4; geom2++) {
	 geometries[geom2] = new GeomCappedCylinder;
	 face->add(geometries[geom2]);
      }
      picks[pick] = new GeomPick(face, module, this, pick);
      picks[pick]->set_highlight(HighlightMaterial);
      resizes->add(picks[pick]);
   }
   GeomMaterial* resizem = new GeomMaterial(resizes, ResizeMaterial);
   CreateModeSwitch(2, resizem);

   SetMode(Mode0, Switch0|Switch1|Switch2);
   SetMode(Mode1, Switch0|Switch1);
   SetMode(Mode2, Switch0|Switch2);
   SetMode(Mode3, Switch0);

   FinishWidget();
}


BoxWidget::~BoxWidget()
{
}


void
BoxWidget::widget_execute()
{
   Real spherediam(widget_scale), resizediam(0.75*widget_scale), cylinderdiam(0.5*widget_scale);
   Point IUL(variables[PointIULVar]->point());
   Point IUR(variables[PointIURVar]->point());
   Point IDR(variables[PointIDRVar]->point());
   Point IDL(variables[PointIDLVar]->point());
   Point OUL(variables[PointOULVar]->point());
   Point OUR(variables[PointOURVar]->point());
   Point ODR(variables[PointODRVar]->point());
   Point ODL(variables[PointODLVar]->point());
   
   if (mode_switches[0]->get_state()) {
      ((GeomCylinder*)geometries[CylIU])->move(IUL, IUR, cylinderdiam);
      ((GeomCylinder*)geometries[CylIR])->move(IUR, IDR, cylinderdiam);
      ((GeomCylinder*)geometries[CylID])->move(IDR, IDL, cylinderdiam);
      ((GeomCylinder*)geometries[CylIL])->move(IDL, IUL, cylinderdiam);
      ((GeomCylinder*)geometries[CylMU])->move(IUL, OUL, cylinderdiam);
      ((GeomCylinder*)geometries[CylMR])->move(IUR, OUR, cylinderdiam);
      ((GeomCylinder*)geometries[CylMD])->move(IDR, ODR, cylinderdiam);
      ((GeomCylinder*)geometries[CylML])->move(IDL, ODL, cylinderdiam);
      ((GeomCylinder*)geometries[CylOU])->move(OUL, OUR, cylinderdiam);
      ((GeomCylinder*)geometries[CylOR])->move(OUR, ODR, cylinderdiam);
      ((GeomCylinder*)geometries[CylOD])->move(ODR, ODL, cylinderdiam);
      ((GeomCylinder*)geometries[CylOL])->move(ODL, OUL, cylinderdiam);
      ((GeomSphere*)geometries[SmallSphereIUL])->move(IUL, cylinderdiam);
      ((GeomSphere*)geometries[SmallSphereIUR])->move(IUR, cylinderdiam);
      ((GeomSphere*)geometries[SmallSphereIDR])->move(IDR, cylinderdiam);
      ((GeomSphere*)geometries[SmallSphereIDL])->move(IDL, cylinderdiam);
      ((GeomSphere*)geometries[SmallSphereOUL])->move(OUL, cylinderdiam);
      ((GeomSphere*)geometries[SmallSphereOUR])->move(OUR, cylinderdiam);
      ((GeomSphere*)geometries[SmallSphereODR])->move(ODR, cylinderdiam);
      ((GeomSphere*)geometries[SmallSphereODL])->move(ODL, cylinderdiam);
   }

   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[SphereIUL])->move(IUL, spherediam);
      ((GeomSphere*)geometries[SphereIUR])->move(IUR, spherediam);
      ((GeomSphere*)geometries[SphereIDR])->move(IDR, spherediam);
      ((GeomSphere*)geometries[SphereIDL])->move(IDL, spherediam);
      ((GeomSphere*)geometries[SphereOUL])->move(OUL, spherediam);
      ((GeomSphere*)geometries[SphereOUR])->move(OUR, spherediam);
      ((GeomSphere*)geometries[SphereODR])->move(ODR, spherediam);
      ((GeomSphere*)geometries[SphereODL])->move(ODL, spherediam);
   }

   if (mode_switches[2]->get_state()) {
      Vector resizelen1(GetAxis1()*0.6*widget_scale),
	 resizelen2(GetAxis2()*0.6*widget_scale),
	 resizelen3(GetAxis3()*0.6*widget_scale);
      
      Point p(OUL + (OUR - OUL) / 3.0);
      ((GeomCappedCylinder*)geometries[GeomResizeUU])->move(p-resizelen2, p+resizelen2, resizediam);
      p = OUR + (IUR - OUR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeUR])->move(p-resizelen2, p+resizelen2, resizediam);
      p = IUR + (IUL - IUR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeUD])->move(p-resizelen2, p+resizelen2, resizediam);
      p = IUL + (OUL - IUL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeUL])->move(p-resizelen2, p+resizelen2, resizediam);
      p = IUR + (OUR - IUR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeRU])->move(p-resizelen1, p+resizelen1, resizediam);
      p = OUR + (ODR - OUR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeRR])->move(p-resizelen1, p+resizelen1, resizediam);
      p = ODR + (IDR - ODR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeRD])->move(p-resizelen1, p+resizelen1, resizediam);
      p = IDR + (IUR - IDR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeRL])->move(p-resizelen1, p+resizelen1, resizediam);
      p = IDL + (IDR - IDL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeDU])->move(p-resizelen2, p+resizelen2, resizediam);
      p = IDR + (ODR - IDR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeDR])->move(p-resizelen2, p+resizelen2, resizediam);
      p = ODR + (ODL - ODR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeDD])->move(p-resizelen2, p+resizelen2, resizediam);
      p = ODL + (IDL - ODL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeDL])->move(p-resizelen2, p+resizelen2, resizediam);
      p = OUL + (IUL - OUL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeLU])->move(p-resizelen1, p+resizelen1, resizediam);
      p = IUL + (IDL - IUL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeLR])->move(p-resizelen1, p+resizelen1, resizediam);
      p = IDL + (ODL - IDL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeLD])->move(p-resizelen1, p+resizelen1, resizediam);
      p = ODL + (OUL - ODL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeLL])->move(p-resizelen1, p+resizelen1, resizediam);
      p = IUL + (IUR - IUL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeIU])->move(p-resizelen3, p+resizelen3, resizediam);
      p = IUR + (IDR - IUR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeIR])->move(p-resizelen3, p+resizelen3, resizediam);
      p = IDR + (IDL - IDR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeID])->move(p-resizelen3, p+resizelen3, resizediam);
      p = IDL + (IUL - IDL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeIL])->move(p-resizelen3, p+resizelen3, resizediam);
      p = OUR + (OUL - OUR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeOU])->move(p-resizelen3, p+resizelen3, resizediam);
      p = OUL + (ODL - OUL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeOR])->move(p-resizelen3, p+resizelen3, resizediam);
      p = ODL + (ODR - ODL) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeOD])->move(p-resizelen3, p+resizelen3, resizediam);
      p = ODR + (OUR - ODR) / 3.0;
      ((GeomCappedCylinder*)geometries[GeomResizeOL])->move(p-resizelen3, p+resizelen3, resizediam);
   }

   Vector spvec1(IUR - IUL), spvec2(IDL - IUL), spvec3(OUL - IUL);
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0) && (spvec3.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      spvec3.normalize();
      for (Index geom = 0; geom < NumPcks; geom++) {
	 picks[geom]->set_principal(spvec1, spvec2, spvec3);
      }
   } else if ((spvec2.length2() > 0.0) && (spvec3.length2() > 0.0)) {
      spvec2.normalize();
      spvec3.normalize();
      Vector v = Cross(spvec2, spvec3);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 picks[geom]->set_principal(v, spvec2, spvec3);
      }
   } else if ((spvec1.length2() > 0.0) && (spvec3.length2() > 0.0)) {
      spvec1.normalize();
      spvec3.normalize();
      Vector v = Cross(spvec1, spvec3);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 picks[geom]->set_principal(spvec1, v, spvec3);
      }
   } else if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
BoxWidget::geom_moved( int /* axis*/, double /*dist*/, const Vector& delta,
		       int cbdata )
{
   switch(cbdata){
   case PickSphIUL:
      variables[PointIULVar]->SetDelta(delta);
      break;
   case PickSphIUR:
      variables[PointIURVar]->SetDelta(delta);
      break;
   case PickSphIDR:
      variables[PointIDRVar]->SetDelta(delta);
      break;
   case PickSphIDL:
      variables[PointIDLVar]->SetDelta(delta);
      break;
   case PickSphOUL:
      variables[PointOULVar]->SetDelta(delta);
      break;
   case PickSphOUR:
      variables[PointOURVar]->SetDelta(delta);
      break;
   case PickSphODR:
      variables[PointODRVar]->SetDelta(delta);
      break;
   case PickSphODL:
      variables[PointODLVar]->SetDelta(delta);
      break;
   case PickResizeU:
      variables[PointOULVar]->MoveDelta(delta);
      variables[PointOURVar]->MoveDelta(delta);
      variables[PointIURVar]->MoveDelta(delta);
      variables[PointIULVar]->SetDelta(delta, Scheme4);
      break;
   case PickResizeR:
      variables[PointIURVar]->MoveDelta(delta);
      variables[PointOURVar]->MoveDelta(delta);
      variables[PointODRVar]->MoveDelta(delta);
      variables[PointIDRVar]->SetDelta(delta, Scheme4);
      break;
   case PickResizeD:
      variables[PointIDLVar]->MoveDelta(delta);
      variables[PointIDRVar]->MoveDelta(delta);
      variables[PointODRVar]->MoveDelta(delta);
      variables[PointODLVar]->SetDelta(delta, Scheme4);
      break;
   case PickResizeL:
      variables[PointOULVar]->MoveDelta(delta);
      variables[PointIULVar]->MoveDelta(delta);
      variables[PointIDLVar]->MoveDelta(delta);
      variables[PointODLVar]->SetDelta(delta, Scheme4);
      break;
   case PickResizeI:
      variables[PointIULVar]->MoveDelta(delta);
      variables[PointIURVar]->MoveDelta(delta);
      variables[PointIDRVar]->MoveDelta(delta);
      variables[PointIDLVar]->SetDelta(delta, Scheme4);
      break;
   case PickResizeO:
      variables[PointOURVar]->MoveDelta(delta);
      variables[PointOULVar]->MoveDelta(delta);
      variables[PointODLVar]->MoveDelta(delta);
      variables[PointODRVar]->SetDelta(delta, Scheme4);
      break;
   case PickCyls:
      MoveDelta(delta);
      break;
   }
   execute();
}


void
BoxWidget::MoveDelta( const Vector& delta )
{
   variables[PointIULVar]->MoveDelta(delta);
   variables[PointIURVar]->MoveDelta(delta);
   variables[PointIDRVar]->MoveDelta(delta);
   variables[PointIDLVar]->MoveDelta(delta);
   variables[PointOULVar]->MoveDelta(delta);
   variables[PointOURVar]->MoveDelta(delta);
   variables[PointODRVar]->MoveDelta(delta);
   variables[PointODLVar]->MoveDelta(delta);

   execute();
}


Point
BoxWidget::ReferencePoint() const
{
   return (variables[PointIULVar]->point()
	   + (variables[PointODRVar]->point()
	      -variables[PointIULVar]->point())/2.0);
}


Vector
BoxWidget::GetAxis1()
{
   Vector axis(variables[PointIURVar]->point() - variables[PointIULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


Vector
BoxWidget::GetAxis2()
{
   Vector axis(variables[PointIDLVar]->point() - variables[PointIULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


Vector
BoxWidget::GetAxis3()
{
   Vector axis(variables[PointOULVar]->point() - variables[PointIULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis3;
   else
      return (oldaxis3 = axis.normal());
}


