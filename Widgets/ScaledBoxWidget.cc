
/*
 *  ScaledBoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/ScaledBoxWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Constraints/SegmentConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 27;
const Index NumVars = 22;
const Index NumGeoms = 55;
const Index NumPcks = 18;
const Index NumMdes = 5;
const Index NumSwtchs = 4;
const Index NumSchemes = 4;

enum { ConstIULODR, ConstOULIDR, ConstIDLOUR, ConstODLIUR,
       ConstHypo, ConstDiag,
       ConstIULUR, ConstIULDL, ConstIDRUR, ConstIDRDL,
       ConstMULUL, ConstMURUR, ConstMDRDR, ConstMDLDL,
       ConstOULUR, ConstOULDL, ConstODRUR, ConstODRDL,
       ConstLine1, ConstSDist1, ConstRatio1,
       ConstLine2, ConstSDist2, ConstRatio2,
       ConstLine3, ConstSDist3, ConstRatio3 };
enum { SmallSphereIUL, SmallSphereIUR, SmallSphereIDR, SmallSphereIDL,
       SmallSphereOUL, SmallSphereOUR, SmallSphereODR, SmallSphereODL,
       SphereIUL, SphereIUR, SphereIDR, SphereIDL,
       SphereOUL, SphereOUR, SphereODR, SphereODL,
       CylIU, CylIR, CylID, CylIL,
       CylMU, CylMR, CylMD, CylML,
       CylOU, CylOR, CylOD, CylOL,
       GeomResizeUU, GeomResizeUR, GeomResizeUD, GeomResizeUL,
       GeomResizeRU, GeomResizeRR, GeomResizeRD, GeomResizeRL,
       GeomResizeDU, GeomResizeDR, GeomResizeDD, GeomResizeDL,
       GeomResizeLU, GeomResizeLR, GeomResizeLD, GeomResizeLL,
       GeomResizeIU, GeomResizeIR, GeomResizeID, GeomResizeIL,
       GeomResizeOU, GeomResizeOR, GeomResizeOD, GeomResizeOL,
       SliderCyl1, SliderCyl2, SliderCyl3 };
enum { PickSphIUL, PickSphIUR, PickSphIDR, PickSphIDL,
       PickSphOUL, PickSphOUR, PickSphODR, PickSphODL,
       PickCyls, PickResizeU, PickResizeR, PickResizeD,
       PickResizeL, PickResizeI, PickResizeO,
       PickSlider1, PickSlider2, PickSlider3 };

ScaledBoxWidget::ScaledBoxWidget( Module* module, CrowdMonitor* lock, double widget_scale )
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
   variables[Slider1Var] = new PointVariable("Slider1", solve, Scheme3, Point(INIT/2.0, 0, 0));
   variables[Slider2Var] = new PointVariable("Slider2", solve, Scheme4, Point(0, INIT/2.0, 0));
   variables[Slider3Var] = new PointVariable("Slider3", solve, Scheme4, Point(0, 0, INIT/2.0));
   variables[Dist1Var] = new RealVariable("DIST1", solve, Scheme1, INIT);
   variables[Dist2Var] = new RealVariable("DIST2", solve, Scheme1, INIT);
   variables[Dist3Var] = new RealVariable("DIST3", solve, Scheme1, INIT);
   variables[HypoVar] = new RealVariable("HYPO", solve, Scheme1, sqrt(2*INIT*INIT));
   variables[DiagVar] = new RealVariable("DIAG", solve, Scheme1, sqrt(3*INIT*INIT));
   variables[SDist1Var] = new RealVariable("SDist1", solve, Scheme3, INIT/2.0);
   variables[SDist2Var] = new RealVariable("SDist2", solve, Scheme4, INIT/2.0);
   variables[SDist3Var] = new RealVariable("SDist3", solve, Scheme4, INIT/2.0);
   variables[Ratio1Var] = new RealVariable("Ratio1", solve, Scheme1, 0.5);
   variables[Ratio2Var] = new RealVariable("Ratio2", solve, Scheme1, 0.5);
   variables[Ratio3Var] = new RealVariable("Ratio3", solve, Scheme1, 0.5);

   NOT_FINISHED("Constraints not right!");
   
   constraints[ConstLine1] = new SegmentConstraint("ConstLine1",
						   NumSchemes,
						   variables[PointOULVar],
						   variables[PointOURVar],
						   variables[Slider1Var]);
   constraints[ConstLine1]->VarChoices(Scheme1, 2, 2, 2);
   constraints[ConstLine1]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstLine1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstLine1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstLine1]->Priorities(P_Default, P_Default, P_Highest);
   constraints[ConstLine2] = new SegmentConstraint("ConstLine2",
						   NumSchemes,
						   variables[PointOULVar],
						   variables[PointODLVar],
						   variables[Slider2Var]);
   constraints[ConstLine2]->VarChoices(Scheme1, 2, 2, 2);
   constraints[ConstLine2]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstLine2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstLine2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstLine2]->Priorities(P_Default, P_Default, P_Highest);
   constraints[ConstLine3] = new SegmentConstraint("ConstLine3",
						   NumSchemes,
						   variables[PointOULVar],
						   variables[PointIULVar],
						   variables[Slider3Var]);
   constraints[ConstLine3]->VarChoices(Scheme1, 2, 2, 2);
   constraints[ConstLine3]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstLine3]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstLine3]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstLine3]->Priorities(P_Default, P_Default, P_Highest);
   constraints[ConstSDist1] = new DistanceConstraint("ConstSDist1",
						     NumSchemes,
						     variables[PointOULVar],
						     variables[Slider1Var],
						     variables[SDist1Var]);
   constraints[ConstSDist1]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstSDist1]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstSDist1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstSDist1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstSDist1]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[ConstRatio1] = new RatioConstraint("ConstRatio1",
						  NumSchemes,
						  variables[SDist1Var],
						  variables[Dist1Var],
						  variables[Ratio1Var]);
   constraints[ConstRatio1]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatio1]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRatio1]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstRatio1]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstRatio1]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstSDist2] = new DistanceConstraint("ConstSDist2",
						     NumSchemes,
						     variables[PointOULVar],
						     variables[Slider2Var],
						     variables[SDist2Var]);
   constraints[ConstSDist2]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstSDist2]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstSDist2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstSDist2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstSDist2]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[ConstRatio2] = new RatioConstraint("ConstRatio2",
						  NumSchemes,
						  variables[SDist2Var],
						  variables[Dist2Var],
						  variables[Ratio2Var]);
   constraints[ConstRatio2]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatio2]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRatio2]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstRatio2]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstRatio2]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstSDist3] = new DistanceConstraint("ConstSDist3",
						     NumSchemes,
						     variables[PointOULVar],
						     variables[Slider3Var],
						     variables[SDist3Var]);
   constraints[ConstSDist3]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstSDist3]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstSDist3]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstSDist3]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstSDist3]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[ConstRatio3] = new RatioConstraint("ConstRatio3",
						  NumSchemes,
						  variables[SDist3Var],
						  variables[Dist3Var],
						  variables[Ratio3Var]);
   constraints[ConstRatio3]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatio3]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRatio3]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstRatio3]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstRatio3]->Priorities(P_Highest, P_Highest, P_Highest);
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
						     variables[Dist1Var],
						     variables[Dist2Var],
						     variables[HypoVar]);
   constraints[ConstHypo]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstHypo]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstHypo]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstDiag] = new PythagorasConstraint("ConstDiag",
						     NumSchemes,
						     variables[Dist3Var],
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
						    variables[Dist1Var]);
   constraints[ConstIULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstIULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstIULDL] = new DistanceConstraint("ConstIULDL",
						    NumSchemes,
						    variables[PointIULVar],
						    variables[PointIDLVar],
						    variables[Dist2Var]);
   constraints[ConstIULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstIULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstIDRUR] = new DistanceConstraint("ConstIDRUR",
						    NumSchemes,
						    variables[PointIDRVar],
						    variables[PointIURVar],
						    variables[Dist2Var]);
   constraints[ConstIDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstIDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstIDRDL] = new DistanceConstraint("ConstIDRDL",
						    NumSchemes,
						    variables[PointIDRVar],
						    variables[PointIDLVar],
						    variables[Dist1Var]);
   constraints[ConstIDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstIDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIDRDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstMULUL] = new DistanceConstraint("ConstMULUL",
						    NumSchemes,
						    variables[PointIULVar],
						    variables[PointOULVar],
						    variables[Dist3Var]);
   constraints[ConstMULUL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstMULUL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstMULUL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstMULUL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstMULUL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstMURUR] = new DistanceConstraint("ConstMURUR",
						    NumSchemes,
						    variables[PointIURVar],
						    variables[PointOURVar],
						    variables[Dist3Var]);
   constraints[ConstMURUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstMURUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstMURUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstMURUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstMURUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstMDRDR] = new DistanceConstraint("ConstMDRDR",
						    NumSchemes,
						    variables[PointIDRVar],
						    variables[PointODRVar],
						    variables[Dist3Var]);
   constraints[ConstMDRDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstMDRDR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstMDRDR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstMDRDR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstMDRDR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstMDLDL] = new DistanceConstraint("ConstMDLDL",
						    NumSchemes,
						    variables[PointIDLVar],
						    variables[PointODLVar],
						    variables[Dist3Var]);
   constraints[ConstMDLDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstMDLDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstMDLDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstMDLDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstMDLDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstOULUR] = new DistanceConstraint("ConstOULUR",
						    NumSchemes,
						    variables[PointOULVar],
						    variables[PointOURVar],
						    variables[Dist1Var]);
   constraints[ConstOULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstOULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstOULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstOULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstOULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstOULDL] = new DistanceConstraint("ConstOULDL",
						    NumSchemes,
						    variables[PointOULVar],
						    variables[PointODLVar],
						    variables[Dist2Var]);
   constraints[ConstOULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstOULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstOULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstOULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstOULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstODRUR] = new DistanceConstraint("ConstODRUR",
						    NumSchemes,
						    variables[PointODRVar],
						    variables[PointOURVar],
						    variables[Dist2Var]);
   constraints[ConstODRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstODRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstODRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstODRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstODRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstODRDL] = new DistanceConstraint("ConstODRDL",
						    NumSchemes,
						    variables[PointODRVar],
						    variables[PointODLVar],
						    variables[Dist1Var]);
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
   picks[PickCyls] = new GeomPick(cyls, module);
   picks[PickCyls]->set_highlight(HighlightMaterial);
   picks[PickCyls]->set_cbdata((void*)PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[PickCyls], EdgeMaterial);
   CreateModeSwitch(0, cylsm);

   GeomGroup* pts = new GeomGroup;
   for (geom = SphereIUL, pick = PickSphIUL;
	geom <= SphereODL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(HighlightMaterial);
      picks[pick]->set_cbdata((void*)pick);
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
      picks[pick] = new GeomPick(face, module);
      picks[pick]->set_highlight(HighlightMaterial);
      picks[pick]->set_cbdata((void*)pick);
      resizes->add(picks[pick]);
   }
   GeomMaterial* resizem = new GeomMaterial(resizes, ResizeMaterial);
   CreateModeSwitch(2, resizem);

   GeomGroup* sliders = new GeomGroup;
   geometries[SliderCyl1] = new GeomCappedCylinder;
   picks[PickSlider1] = new GeomPick(geometries[SliderCyl1], module);
   picks[PickSlider1]->set_highlight(HighlightMaterial);
   picks[PickSlider1]->set_cbdata((void*)PickSlider1);
   sliders->add(picks[PickSlider1]);
   geometries[SliderCyl2] = new GeomCappedCylinder;
   picks[PickSlider2] = new GeomPick(geometries[SliderCyl2], module);
   picks[PickSlider2]->set_highlight(HighlightMaterial);
   picks[PickSlider2]->set_cbdata((void*)PickSlider2);
   sliders->add(picks[PickSlider2]);
   geometries[SliderCyl3] = new GeomCappedCylinder;
   picks[PickSlider3] = new GeomPick(geometries[SliderCyl3], module);
   picks[PickSlider3]->set_highlight(HighlightMaterial);
   picks[PickSlider3]->set_cbdata((void*)PickSlider3);
   sliders->add(picks[PickSlider3]);
   GeomMaterial* slidersm = new GeomMaterial(sliders, SliderMaterial);
   CreateModeSwitch(3, slidersm);

   SetMode(Mode1, Switch0|Switch1|Switch2|Switch3);
   SetMode(Mode2, Switch0|Switch1|Switch3);
   SetMode(Mode3, Switch0|Switch2|Switch3);
   SetMode(Mode4, Switch0|Switch3);
   SetMode(Mode5, Switch0);

   SetEpsilon(widget_scale*1e-6);

   FinishWidget();
}


ScaledBoxWidget::~ScaledBoxWidget()
{
}


void
ScaledBoxWidget::widget_execute()
{
   ((GeomSphere*)geometries[SmallSphereIUL])->move(variables[PointIULVar]->point(),
						   0.5*widget_scale);
   ((GeomSphere*)geometries[SmallSphereIUR])->move(variables[PointIURVar]->point(),
						   0.5*widget_scale);
   ((GeomSphere*)geometries[SmallSphereIDR])->move(variables[PointIDRVar]->point(),
						   0.5*widget_scale);
   ((GeomSphere*)geometries[SmallSphereIDL])->move(variables[PointIDLVar]->point(),
						   0.5*widget_scale);
   ((GeomSphere*)geometries[SmallSphereOUL])->move(variables[PointOULVar]->point(),
						   0.5*widget_scale);
   ((GeomSphere*)geometries[SmallSphereOUR])->move(variables[PointOURVar]->point(),
						   0.5*widget_scale);
   ((GeomSphere*)geometries[SmallSphereODR])->move(variables[PointODRVar]->point(),
						   0.5*widget_scale);
   ((GeomSphere*)geometries[SmallSphereODL])->move(variables[PointODLVar]->point(),
						   0.5*widget_scale);
   ((GeomSphere*)geometries[SphereIUL])->move(variables[PointIULVar]->point(),
					      1*widget_scale);
   ((GeomSphere*)geometries[SphereIUR])->move(variables[PointIURVar]->point(),
					      1*widget_scale);
   ((GeomSphere*)geometries[SphereIDR])->move(variables[PointIDRVar]->point(),
					      1*widget_scale);
   ((GeomSphere*)geometries[SphereIDL])->move(variables[PointIDLVar]->point(),
					      1*widget_scale);
   ((GeomSphere*)geometries[SphereOUL])->move(variables[PointOULVar]->point(),
					      1*widget_scale);
   ((GeomSphere*)geometries[SphereOUR])->move(variables[PointOURVar]->point(),
					      1*widget_scale);
   ((GeomSphere*)geometries[SphereODR])->move(variables[PointODRVar]->point(),
					      1*widget_scale);
   ((GeomSphere*)geometries[SphereODL])->move(variables[PointODLVar]->point(),
					      1*widget_scale);
   Point p(variables[PointOULVar]->point() + (variables[PointOURVar]->point()
						 - variables[PointOULVar]->point()) / 3.0);
   ((GeomCappedCylinder*)geometries[GeomResizeUU])->move(p - (GetAxis2() * 0.6 * widget_scale),
							 p + (GetAxis2() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointOURVar]->point() + (variables[PointIURVar]->point()
					     - variables[PointOURVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeUR])->move(p - (GetAxis2() * 0.6 * widget_scale),
							 p + (GetAxis2() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIURVar]->point() + (variables[PointIULVar]->point()
					     - variables[PointIURVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeUD])->move(p - (GetAxis2() * 0.6 * widget_scale),
							 p + (GetAxis2() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIULVar]->point() + (variables[PointOULVar]->point()
					     - variables[PointIULVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeUL])->move(p - (GetAxis2() * 0.6 * widget_scale),
							 p + (GetAxis2() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIURVar]->point() + (variables[PointOURVar]->point()
					     - variables[PointIURVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeRU])->move(p - (GetAxis1() * 0.6 * widget_scale),
							 p + (GetAxis1() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointOURVar]->point() + (variables[PointODRVar]->point()
					     - variables[PointOURVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeRR])->move(p - (GetAxis1() * 0.6 * widget_scale),
							 p + (GetAxis1() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointODRVar]->point() + (variables[PointIDRVar]->point()
					     - variables[PointODRVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeRD])->move(p - (GetAxis1() * 0.6 * widget_scale),
							 p + (GetAxis1() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIDRVar]->point() + (variables[PointIURVar]->point()
					     - variables[PointIDRVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeRL])->move(p - (GetAxis1() * 0.6 * widget_scale),
							 p + (GetAxis1() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIDLVar]->point() + (variables[PointIDRVar]->point()
					     - variables[PointIDLVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeDU])->move(p - (GetAxis2() * 0.6 * widget_scale),
							 p + (GetAxis2() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIDRVar]->point() + (variables[PointODRVar]->point()
					     - variables[PointIDRVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeDR])->move(p - (GetAxis2() * 0.6 * widget_scale),
							 p + (GetAxis2() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointODRVar]->point() + (variables[PointODLVar]->point()
					     - variables[PointODRVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeDD])->move(p - (GetAxis2() * 0.6 * widget_scale),
							 p + (GetAxis2() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointODLVar]->point() + (variables[PointIDLVar]->point()
					     - variables[PointODLVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeDL])->move(p - (GetAxis2() * 0.6 * widget_scale),
							 p + (GetAxis2() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointOULVar]->point() + (variables[PointIULVar]->point()
					     - variables[PointOULVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeLU])->move(p - (GetAxis1() * 0.6 * widget_scale),
							 p + (GetAxis1() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIULVar]->point() + (variables[PointIDLVar]->point()
					     - variables[PointIULVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeLR])->move(p - (GetAxis1() * 0.6 * widget_scale),
							 p + (GetAxis1() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIDLVar]->point() + (variables[PointODLVar]->point()
					     - variables[PointIDLVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeLD])->move(p - (GetAxis1() * 0.6 * widget_scale),
							 p + (GetAxis1() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointODLVar]->point() + (variables[PointOULVar]->point()
					     - variables[PointODLVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeLL])->move(p - (GetAxis1() * 0.6 * widget_scale),
							 p + (GetAxis1() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIULVar]->point() + (variables[PointIURVar]->point()
					     - variables[PointIULVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeIU])->move(p - (GetAxis3() * 0.6 * widget_scale),
							 p + (GetAxis3() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIURVar]->point() + (variables[PointIDRVar]->point()
					     - variables[PointIURVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeIR])->move(p - (GetAxis3() * 0.6 * widget_scale),
							 p + (GetAxis3() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIDRVar]->point() + (variables[PointIDLVar]->point()
					     - variables[PointIDRVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeID])->move(p - (GetAxis3() * 0.6 * widget_scale),
							 p + (GetAxis3() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointIDLVar]->point() + (variables[PointIULVar]->point()
					     - variables[PointIDLVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeIL])->move(p - (GetAxis3() * 0.6 * widget_scale),
							 p + (GetAxis3() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointOURVar]->point() + (variables[PointOULVar]->point()
					     - variables[PointOURVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeOU])->move(p - (GetAxis3() * 0.6 * widget_scale),
							 p + (GetAxis3() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointOULVar]->point() + (variables[PointODLVar]->point()
					     - variables[PointOULVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeOR])->move(p - (GetAxis3() * 0.6 * widget_scale),
							 p + (GetAxis3() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointODLVar]->point() + (variables[PointODRVar]->point()
					     - variables[PointODLVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeOD])->move(p - (GetAxis3() * 0.6 * widget_scale),
							 p + (GetAxis3() * 0.6 * widget_scale),
							 0.75*widget_scale);
   p = variables[PointODRVar]->point() + (variables[PointOURVar]->point()
					     - variables[PointODRVar]->point()) / 3.0;
   ((GeomCappedCylinder*)geometries[GeomResizeOL])->move(p - (GetAxis3() * 0.6 * widget_scale),
							 p + (GetAxis3() * 0.6 * widget_scale),
							 0.75*widget_scale);
   ((GeomCylinder*)geometries[CylIU])->move(variables[PointIULVar]->point(),
					    variables[PointIURVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylIR])->move(variables[PointIURVar]->point(),
					    variables[PointIDRVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylID])->move(variables[PointIDRVar]->point(),
					    variables[PointIDLVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylIL])->move(variables[PointIDLVar]->point(),
					    variables[PointIULVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylMU])->move(variables[PointIULVar]->point(),
					    variables[PointOULVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylMR])->move(variables[PointIURVar]->point(),
					    variables[PointOURVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylMD])->move(variables[PointIDRVar]->point(),
					    variables[PointODRVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylML])->move(variables[PointIDLVar]->point(),
					    variables[PointODLVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylOU])->move(variables[PointOULVar]->point(),
					    variables[PointOURVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylOR])->move(variables[PointOURVar]->point(),
					    variables[PointODRVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylOD])->move(variables[PointODRVar]->point(),
					    variables[PointODLVar]->point(),
					    0.5*widget_scale);
   ((GeomCylinder*)geometries[CylOL])->move(variables[PointODLVar]->point(),
					    variables[PointOULVar]->point(),
					    0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[SliderCyl1])->move(variables[Slider1Var]->point()
						       - (GetAxis1() * 0.3 * widget_scale),
						       variables[Slider1Var]->point()
						       + (GetAxis1() * 0.3 * widget_scale),
						       1.1*widget_scale);
   ((GeomCappedCylinder*)geometries[SliderCyl2])->move(variables[Slider2Var]->point()
						       - (GetAxis2() * 0.3 * widget_scale),
						       variables[Slider2Var]->point()
						       + (GetAxis2() * 0.3 * widget_scale),
						       1.1*widget_scale);
   ((GeomCappedCylinder*)geometries[SliderCyl3])->move(variables[Slider3Var]->point()
						       - (GetAxis3() * 0.3 * widget_scale),
						       variables[Slider3Var]->point()
						       + (GetAxis3() * 0.3 * widget_scale),
						       1.1*widget_scale);

   SetEpsilon(widget_scale*1e-6);

   Vector spvec1(variables[PointIURVar]->point() - variables[PointIULVar]->point());
   Vector spvec2(variables[PointIDLVar]->point() - variables[PointIULVar]->point());
   Vector spvec3(variables[PointOULVar]->point() - variables[PointIULVar]->point());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0) && (spvec3.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      spvec3.normalize();
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == PickSlider3)
	    picks[geom]->set_principal(spvec3);
	 else if (geom == PickSlider2)
	    picks[geom]->set_principal(spvec2);
	 else if (geom == PickSlider1)
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, spvec3);
      }
   } else if ((spvec2.length2() > 0.0) && (spvec3.length2() > 0.0)) {
      spvec2.normalize();
      spvec3.normalize();
      Vector v = Cross(spvec2, spvec3);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == PickSlider3)
	    picks[geom]->set_principal(spvec3);
	 else if (geom == PickSlider2)
	    picks[geom]->set_principal(spvec2);
	 else if (geom == PickSlider1)
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(v, spvec2, spvec3);
      }
   } else if ((spvec1.length2() > 0.0) && (spvec3.length2() > 0.0)) {
      spvec1.normalize();
      spvec3.normalize();
      Vector v = Cross(spvec1, spvec3);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == PickSlider3)
	    picks[geom]->set_principal(spvec3);
	 else if (geom == PickSlider2)
	    picks[geom]->set_principal(spvec2);
	 else if (geom == PickSlider1)
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(spvec1, v, spvec3);
      }
   } else if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == PickSlider3)
	    picks[geom]->set_principal(spvec3);
	 else if (geom == PickSlider2)
	    picks[geom]->set_principal(spvec2);
	 else if (geom == PickSlider1)
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
ScaledBoxWidget::geom_moved( int /* axis*/, double /*dist*/, const Vector& delta,
			     void* cbdata )
{
   switch((int)cbdata){
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
   case PickSlider1:
      variables[Slider1Var]->SetDelta(delta);
      break;
   case PickSlider2:
      variables[Slider2Var]->SetDelta(delta);
      break;
   case PickSlider3:
      variables[Slider3Var]->SetDelta(delta);
      break;
   case PickCyls:
      MoveDelta(delta);
      break;
   }
}


void
ScaledBoxWidget::MoveDelta( const Vector& delta )
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
ScaledBoxWidget::ReferencePoint() const
{
   return (variables[PointIULVar]->point()
	   + (variables[PointODRVar]->point()
	      -variables[PointIULVar]->point())/2.0);
}


Real
ScaledBoxWidget::GetRatio1() const
{
   return (variables[Ratio1Var]->real());
}


Real
ScaledBoxWidget::GetRatio2() const
{
   return (variables[Ratio2Var]->real());
}


Real
ScaledBoxWidget::GetRatio3() const
{
   return (variables[Ratio3Var]->real());
}


Vector
ScaledBoxWidget::GetAxis1()
{
   Vector axis(variables[PointIURVar]->point() - variables[PointIULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


Vector
ScaledBoxWidget::GetAxis2()
{
   Vector axis(variables[PointIDLVar]->point() - variables[PointIULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


Vector
ScaledBoxWidget::GetAxis3()
{
   Vector axis(variables[PointOULVar]->point() - variables[PointIULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis3;
   else
      return (oldaxis3 = axis.normal());
}


