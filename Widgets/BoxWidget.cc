
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
const Index NumGeoms = 44;
const Index NumMatls = 3;
const Index NumSchemes = 4;
const Index NumPcks = 15;

enum { BoxW_ConstIULODR, BoxW_ConstOULIDR, BoxW_ConstIDLOUR, BoxW_ConstODLIUR,
       BoxW_ConstHypo, BoxW_ConstDiag,
       BoxW_ConstIULUR, BoxW_ConstIULDL, BoxW_ConstIDRUR, BoxW_ConstIDRDL,
       BoxW_ConstMULUL, BoxW_ConstMURUR, BoxW_ConstMDRDR, BoxW_ConstMDLDL,
       BoxW_ConstOULUR, BoxW_ConstOULDL, BoxW_ConstODRUR, BoxW_ConstODRDL };
enum { BoxW_SphereIUL, BoxW_SphereIUR, BoxW_SphereIDR, BoxW_SphereIDL,
       BoxW_SphereOUL, BoxW_SphereOUR, BoxW_SphereODR, BoxW_SphereODL,
       BoxW_CylIU, BoxW_CylIR, BoxW_CylID, BoxW_CylIL,
       BoxW_CylMU, BoxW_CylMR, BoxW_CylMD, BoxW_CylML,
       BoxW_CylOU, BoxW_CylOR, BoxW_CylOD, BoxW_CylOL,
       BoxW_GeomResizeUU, BoxW_GeomResizeUR, BoxW_GeomResizeUD, BoxW_GeomResizeUL,
       BoxW_GeomResizeRU, BoxW_GeomResizeRR, BoxW_GeomResizeRD, BoxW_GeomResizeRL,
       BoxW_GeomResizeDU, BoxW_GeomResizeDR, BoxW_GeomResizeDD, BoxW_GeomResizeDL,
       BoxW_GeomResizeLU, BoxW_GeomResizeLR, BoxW_GeomResizeLD, BoxW_GeomResizeLL,
       BoxW_GeomResizeIU, BoxW_GeomResizeIR, BoxW_GeomResizeID, BoxW_GeomResizeIL,
       BoxW_GeomResizeOU, BoxW_GeomResizeOR, BoxW_GeomResizeOD, BoxW_GeomResizeOL };
enum { BoxW_PickSphIUL, BoxW_PickSphIUR, BoxW_PickSphIDR, BoxW_PickSphIDL,
       BoxW_PickSphOUL, BoxW_PickSphOUR, BoxW_PickSphODR, BoxW_PickSphODL,
       BoxW_PickCyls, BoxW_PickResizeU, BoxW_PickResizeR, BoxW_PickResizeD,
       BoxW_PickResizeL, BoxW_PickResizeI, BoxW_PickResizeO };

BoxWidget::BoxWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale)
{
   Real INIT = 1.0*widget_scale;
   variables[BoxW_PointIUL] = new Variable("PntIUL", Scheme1, Point(0, 0, 0));
   variables[BoxW_PointIUR] = new Variable("PntIUR", Scheme2, Point(INIT, 0, 0));
   variables[BoxW_PointIDR] = new Variable("PntIDR", Scheme1, Point(INIT, INIT, 0));
   variables[BoxW_PointIDL] = new Variable("PntIDL", Scheme2, Point(0, INIT, 0));
   variables[BoxW_PointOUL] = new Variable("PntOUL", Scheme1, Point(0, 0, INIT));
   variables[BoxW_PointOUR] = new Variable("PntOUR", Scheme2, Point(INIT, 0, INIT));
   variables[BoxW_PointODR] = new Variable("PntODR", Scheme1, Point(INIT, INIT, INIT));
   variables[BoxW_PointODL] = new Variable("PntODL", Scheme2, Point(0, INIT, INIT));
   variables[BoxW_Dist] = new Variable("DIST", Scheme1, Point(INIT, 0, 0));
   variables[BoxW_Hypo] = new Variable("HYPO", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));
   variables[BoxW_Diag] = new Variable("DIAG", Scheme1, Point(sqrt(3*INIT*INIT), 0, 0));

   NOT_FINISHED("Constraints not right!");
   
   constraints[BoxW_ConstIULODR] = new DistanceConstraint("ConstIULODR",
							  NumSchemes,
							  variables[BoxW_PointIUL],
							  variables[BoxW_PointODR],
							  variables[BoxW_Diag]);
   constraints[BoxW_ConstIULODR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[BoxW_ConstIULODR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[BoxW_ConstIULODR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[BoxW_ConstIULODR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[BoxW_ConstIULODR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[BoxW_ConstOULIDR] = new DistanceConstraint("ConstOULIDR",
							  NumSchemes,
							  variables[BoxW_PointOUL],
							  variables[BoxW_PointIDR],
							  variables[BoxW_Diag]);
   constraints[BoxW_ConstOULIDR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[BoxW_ConstOULIDR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[BoxW_ConstOULIDR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[BoxW_ConstOULIDR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[BoxW_ConstOULIDR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[BoxW_ConstIDLOUR] = new DistanceConstraint("ConstIDLOUR",
							  NumSchemes,
							  variables[BoxW_PointIDL],
							  variables[BoxW_PointOUR],
							  variables[BoxW_Diag]);
   constraints[BoxW_ConstIDLOUR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[BoxW_ConstIDLOUR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[BoxW_ConstIDLOUR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[BoxW_ConstIDLOUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[BoxW_ConstIDLOUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[BoxW_ConstODLIUR] = new DistanceConstraint("ConstODLIUR",
							  NumSchemes,
							  variables[BoxW_PointODL],
							  variables[BoxW_PointIUR],
							  variables[BoxW_Diag]);
   constraints[BoxW_ConstODLIUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[BoxW_ConstODLIUR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[BoxW_ConstODLIUR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[BoxW_ConstODLIUR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[BoxW_ConstODLIUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[BoxW_ConstHypo] = new PythagorasConstraint("ConstHypo",
							  NumSchemes,
							  variables[BoxW_Dist],
							  variables[BoxW_Dist],
							  variables[BoxW_Hypo]);
   constraints[BoxW_ConstHypo]->VarChoices(Scheme1, 1, 0, 1);
   constraints[BoxW_ConstHypo]->VarChoices(Scheme2, 1, 0, 1);
   constraints[BoxW_ConstHypo]->VarChoices(Scheme3, 1, 0, 1);
   constraints[BoxW_ConstHypo]->VarChoices(Scheme4, 1, 0, 1);
   constraints[BoxW_ConstHypo]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[BoxW_ConstDiag] = new PythagorasConstraint("ConstDiag",
							  NumSchemes,
							  variables[BoxW_Dist],
							  variables[BoxW_Hypo],
							  variables[BoxW_Diag]);
   constraints[BoxW_ConstDiag]->VarChoices(Scheme1, 2, 2, 1);
   constraints[BoxW_ConstDiag]->VarChoices(Scheme2, 2, 2, 1);
   constraints[BoxW_ConstDiag]->VarChoices(Scheme3, 2, 2, 1);
   constraints[BoxW_ConstDiag]->VarChoices(Scheme4, 2, 2, 1);
   constraints[BoxW_ConstDiag]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[BoxW_ConstIULUR] = new DistanceConstraint("ConstIULUR",
							 NumSchemes,
							 variables[BoxW_PointIUL],
							 variables[BoxW_PointIUR],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstIULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstIULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstIULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstIULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstIULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstIULDL] = new DistanceConstraint("ConstIULDL",
							 NumSchemes,
							 variables[BoxW_PointIUL],
							 variables[BoxW_PointIDL],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstIULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstIULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstIULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstIULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstIULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstIDRUR] = new DistanceConstraint("ConstIDRUR",
							 NumSchemes,
							 variables[BoxW_PointIDR],
							 variables[BoxW_PointIUR],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstIDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstIDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstIDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstIDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstIDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstIDRDL] = new DistanceConstraint("ConstIDRUR",
							 NumSchemes,
							 variables[BoxW_PointIDR],
							 variables[BoxW_PointIDL],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstIDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstIDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstIDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstIDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstIDRDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstMULUL] = new DistanceConstraint("ConstMULUL",
							 NumSchemes,
							 variables[BoxW_PointIUL],
							 variables[BoxW_PointOUL],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstMULUL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstMULUL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstMULUL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstMULUL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstMULUL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstMURUR] = new DistanceConstraint("ConstMURUR",
							 NumSchemes,
							 variables[BoxW_PointIUR],
							 variables[BoxW_PointOUR],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstMURUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstMURUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstMURUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstMURUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstMURUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstMDRDR] = new DistanceConstraint("ConstMDRDR",
							 NumSchemes,
							 variables[BoxW_PointIDR],
							 variables[BoxW_PointODR],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstMDRDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstMDRDR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstMDRDR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstMDRDR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstMDRDR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstMDLDL] = new DistanceConstraint("ConstMDLDL",
							 NumSchemes,
							 variables[BoxW_PointIDL],
							 variables[BoxW_PointODL],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstMDLDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstMDLDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstMDLDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstMDLDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstMDLDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstOULUR] = new DistanceConstraint("ConstOULUR",
							 NumSchemes,
							 variables[BoxW_PointOUL],
							 variables[BoxW_PointOUR],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstOULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstOULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstOULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstOULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstOULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstOULDL] = new DistanceConstraint("ConstOULDL",
							 NumSchemes,
							 variables[BoxW_PointOUL],
							 variables[BoxW_PointODL],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstOULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstOULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstOULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstOULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstOULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstODRUR] = new DistanceConstraint("ConstODRUR",
							 NumSchemes,
							 variables[BoxW_PointODR],
							 variables[BoxW_PointOUR],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstODRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstODRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstODRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstODRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstODRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[BoxW_ConstODRDL] = new DistanceConstraint("ConstODRDL",
							 NumSchemes,
							 variables[BoxW_PointODR],
							 variables[BoxW_PointODL],
							 variables[BoxW_Dist]);
   constraints[BoxW_ConstODRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[BoxW_ConstODRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[BoxW_ConstODRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[BoxW_ConstODRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[BoxW_ConstODRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[BoxW_PointMatl] = PointWidgetMaterial;
   materials[BoxW_EdgeMatl] = EdgeWidgetMaterial;
   materials[BoxW_HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = BoxW_SphereIUL, pick = BoxW_PickSphIUL;
	geom <= BoxW_SphereODL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[BoxW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[BoxW_PointMatl]);
   
   GeomGroup* resizes = new GeomGroup;
   GeomGroup* face;
   for (geom = BoxW_GeomResizeUU, pick = BoxW_PickResizeU;
	geom <= BoxW_GeomResizeOL; geom+=4, pick++) {
      face = new GeomGroup;
      for (Index geom2=geom; geom2<geom+4; geom2++) {
	 geometries[geom2] = new GeomCappedCylinder;
	 face->add(geometries[geom2]);
      }
      picks[pick] = new GeomPick(face, module);
      picks[pick]->set_highlight(materials[BoxW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      resizes->add(picks[pick]);
   }
   GeomMaterial* resizem = new GeomMaterial(resizes, materials[BoxW_PointMatl]);

   GeomGroup* cyls = new GeomGroup;
   for (geom = BoxW_CylIU; geom <= BoxW_CylOL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[BoxW_PickCyls] = new GeomPick(cyls, module);
   picks[BoxW_PickCyls]->set_highlight(materials[BoxW_HighMatl]);
   picks[BoxW_PickCyls]->set_cbdata((void*)BoxW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[BoxW_PickCyls], materials[BoxW_EdgeMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(resizem);
   w->add(cylsm);

   SetEpsilon(widget_scale*1e-6);

   FinishWidget(w);
}


BoxWidget::~BoxWidget()
{
}


void
BoxWidget::widget_execute()
{
   ((GeomSphere*)geometries[BoxW_SphereIUL])->move(variables[BoxW_PointIUL]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[BoxW_SphereIUR])->move(variables[BoxW_PointIUR]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[BoxW_SphereIDR])->move(variables[BoxW_PointIDR]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[BoxW_SphereIDL])->move(variables[BoxW_PointIDL]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[BoxW_SphereOUL])->move(variables[BoxW_PointOUL]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[BoxW_SphereOUR])->move(variables[BoxW_PointOUR]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[BoxW_SphereODR])->move(variables[BoxW_PointODR]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[BoxW_SphereODL])->move(variables[BoxW_PointODL]->Get(),
						   1*widget_scale);
   Point p(variables[BoxW_PointOUL]->Get() + (variables[BoxW_PointOUR]->Get()
					     - variables[BoxW_PointOUL]->Get()) / 3.0);
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeUU])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointOUR]->Get() + (variables[BoxW_PointIUR]->Get()
					  - variables[BoxW_PointOUR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeUR])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIUR]->Get() + (variables[BoxW_PointIUL]->Get()
					  - variables[BoxW_PointIUR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeUD])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIUL]->Get() + (variables[BoxW_PointOUL]->Get()
					  - variables[BoxW_PointIUL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeUL])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIUR]->Get() + (variables[BoxW_PointOUR]->Get()
					  - variables[BoxW_PointIUR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeRU])->move(p - (GetAxis1() * 0.6 * widget_scale),
							      p + (GetAxis1() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointOUR]->Get() + (variables[BoxW_PointODR]->Get()
					  - variables[BoxW_PointOUR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeRR])->move(p - (GetAxis1() * 0.6 * widget_scale),
							      p + (GetAxis1() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointODR]->Get() + (variables[BoxW_PointIDR]->Get()
					  - variables[BoxW_PointODR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeRD])->move(p - (GetAxis1() * 0.6 * widget_scale),
							      p + (GetAxis1() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIDR]->Get() + (variables[BoxW_PointIUR]->Get()
					  - variables[BoxW_PointIDR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeRL])->move(p - (GetAxis1() * 0.6 * widget_scale),
							      p + (GetAxis1() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIDL]->Get() + (variables[BoxW_PointIDR]->Get()
					  - variables[BoxW_PointIDL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeDU])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIDR]->Get() + (variables[BoxW_PointODR]->Get()
					  - variables[BoxW_PointIDR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeDR])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointODR]->Get() + (variables[BoxW_PointODL]->Get()
					  - variables[BoxW_PointODR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeDD])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointODL]->Get() + (variables[BoxW_PointIDL]->Get()
					  - variables[BoxW_PointODL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeDL])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointOUL]->Get() + (variables[BoxW_PointIUL]->Get()
					  - variables[BoxW_PointOUL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeLU])->move(p - (GetAxis1() * 0.6 * widget_scale),
							      p + (GetAxis1() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIUL]->Get() + (variables[BoxW_PointIDL]->Get()
					  - variables[BoxW_PointIUL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeLR])->move(p - (GetAxis1() * 0.6 * widget_scale),
							      p + (GetAxis1() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIDL]->Get() + (variables[BoxW_PointODL]->Get()
					  - variables[BoxW_PointIDL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeLD])->move(p - (GetAxis1() * 0.6 * widget_scale),
							      p + (GetAxis1() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointODL]->Get() + (variables[BoxW_PointOUL]->Get()
					  - variables[BoxW_PointODL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeLL])->move(p - (GetAxis1() * 0.6 * widget_scale),
							      p + (GetAxis1() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIUL]->Get() + (variables[BoxW_PointIUR]->Get()
					  - variables[BoxW_PointIUL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeIU])->move(p - (GetAxis3() * 0.6 * widget_scale),
							      p + (GetAxis3() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIUR]->Get() + (variables[BoxW_PointIDR]->Get()
					  - variables[BoxW_PointIUR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeIR])->move(p - (GetAxis3() * 0.6 * widget_scale),
							      p + (GetAxis3() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIDR]->Get() + (variables[BoxW_PointIDL]->Get()
					  - variables[BoxW_PointIDR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeID])->move(p - (GetAxis3() * 0.6 * widget_scale),
							      p + (GetAxis3() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointIDL]->Get() + (variables[BoxW_PointIUL]->Get()
					  - variables[BoxW_PointIDL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeIL])->move(p - (GetAxis3() * 0.6 * widget_scale),
							      p + (GetAxis3() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointOUR]->Get() + (variables[BoxW_PointOUL]->Get()
					  - variables[BoxW_PointOUR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeOU])->move(p - (GetAxis3() * 0.6 * widget_scale),
							      p + (GetAxis3() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointOUL]->Get() + (variables[BoxW_PointODL]->Get()
					  - variables[BoxW_PointOUL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeOR])->move(p - (GetAxis3() * 0.6 * widget_scale),
							      p + (GetAxis3() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointODL]->Get() + (variables[BoxW_PointODR]->Get()
					  - variables[BoxW_PointODL]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeOD])->move(p - (GetAxis3() * 0.6 * widget_scale),
							      p + (GetAxis3() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[BoxW_PointODR]->Get() + (variables[BoxW_PointOUR]->Get()
					  - variables[BoxW_PointODR]->Get()) / 3.0;
   ((GeomCappedCylinder*)geometries[BoxW_GeomResizeOL])->move(p - (GetAxis3() * 0.6 * widget_scale),
							      p + (GetAxis3() * 0.6 * widget_scale),
							      0.75*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylIU])->move(variables[BoxW_PointIUL]->Get(),
						 variables[BoxW_PointIUR]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylIR])->move(variables[BoxW_PointIUR]->Get(),
						 variables[BoxW_PointIDR]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylID])->move(variables[BoxW_PointIDR]->Get(),
						 variables[BoxW_PointIDL]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylIL])->move(variables[BoxW_PointIDL]->Get(),
						 variables[BoxW_PointIUL]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylMU])->move(variables[BoxW_PointIUL]->Get(),
						 variables[BoxW_PointOUL]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylMR])->move(variables[BoxW_PointIUR]->Get(),
						 variables[BoxW_PointOUR]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylMD])->move(variables[BoxW_PointIDR]->Get(),
						 variables[BoxW_PointODR]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylML])->move(variables[BoxW_PointIDL]->Get(),
						 variables[BoxW_PointODL]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylOU])->move(variables[BoxW_PointOUL]->Get(),
						 variables[BoxW_PointOUR]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylOR])->move(variables[BoxW_PointOUR]->Get(),
						 variables[BoxW_PointODR]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylOD])->move(variables[BoxW_PointODR]->Get(),
						 variables[BoxW_PointODL]->Get(),
						 0.5*widget_scale);
   ((GeomCylinder*)geometries[BoxW_CylOL])->move(variables[BoxW_PointODL]->Get(),
						 variables[BoxW_PointOUL]->Get(),
						 0.5*widget_scale);

   SetEpsilon(widget_scale*1e-6);

   Vector spvec1(variables[BoxW_PointIUR]->Get() - variables[BoxW_PointIUL]->Get());
   Vector spvec2(variables[BoxW_PointIDL]->Get() - variables[BoxW_PointIUL]->Get());
   Vector spvec3(variables[BoxW_PointOUL]->Get() - variables[BoxW_PointIUL]->Get());
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
		       void* cbdata )
{
   for (Index v=0; v<NumVars; v++)
      variables[v]->Reset();
   
   switch((int)cbdata){
   case BoxW_PickSphIUL:
      variables[BoxW_PointIUL]->SetDelta(delta);
      break;
   case BoxW_PickSphIUR:
      variables[BoxW_PointIUR]->SetDelta(delta);
      break;
   case BoxW_PickSphIDR:
      variables[BoxW_PointIDR]->SetDelta(delta);
      break;
   case BoxW_PickSphIDL:
      variables[BoxW_PointIDL]->SetDelta(delta);
      break;
   case BoxW_PickSphOUL:
      variables[BoxW_PointOUL]->SetDelta(delta);
      break;
   case BoxW_PickSphOUR:
      variables[BoxW_PointOUR]->SetDelta(delta);
      break;
   case BoxW_PickSphODR:
      variables[BoxW_PointODR]->SetDelta(delta);
      break;
   case BoxW_PickSphODL:
      variables[BoxW_PointODL]->SetDelta(delta);
      break;
   case BoxW_PickResizeU:
      variables[BoxW_PointOUL]->MoveDelta(delta);
      variables[BoxW_PointOUR]->MoveDelta(delta);
      variables[BoxW_PointIUR]->MoveDelta(delta);
      variables[BoxW_PointIUL]->SetDelta(delta, Scheme4);
      break;
   case BoxW_PickResizeR:
      variables[BoxW_PointIUR]->MoveDelta(delta);
      variables[BoxW_PointOUR]->MoveDelta(delta);
      variables[BoxW_PointODR]->MoveDelta(delta);
      variables[BoxW_PointIDR]->SetDelta(delta, Scheme4);
      break;
   case BoxW_PickResizeD:
      variables[BoxW_PointIDL]->MoveDelta(delta);
      variables[BoxW_PointIDR]->MoveDelta(delta);
      variables[BoxW_PointODR]->MoveDelta(delta);
      variables[BoxW_PointODL]->SetDelta(delta, Scheme4);
      break;
   case BoxW_PickResizeL:
      variables[BoxW_PointOUL]->MoveDelta(delta);
      variables[BoxW_PointIUL]->MoveDelta(delta);
      variables[BoxW_PointIDL]->MoveDelta(delta);
      variables[BoxW_PointODL]->SetDelta(delta, Scheme4);
      break;
   case BoxW_PickResizeI:
      variables[BoxW_PointIUL]->MoveDelta(delta);
      variables[BoxW_PointIUR]->MoveDelta(delta);
      variables[BoxW_PointIDR]->MoveDelta(delta);
      variables[BoxW_PointIDL]->SetDelta(delta, Scheme4);
      break;
   case BoxW_PickResizeO:
      variables[BoxW_PointOUR]->MoveDelta(delta);
      variables[BoxW_PointOUL]->MoveDelta(delta);
      variables[BoxW_PointODL]->MoveDelta(delta);
      variables[BoxW_PointODR]->SetDelta(delta, Scheme4);
      break;
   case BoxW_PickCyls:
      variables[BoxW_PointIUL]->MoveDelta(delta);
      variables[BoxW_PointIUR]->MoveDelta(delta);
      variables[BoxW_PointIDR]->MoveDelta(delta);
      variables[BoxW_PointIDL]->MoveDelta(delta);
      variables[BoxW_PointOUL]->MoveDelta(delta);
      variables[BoxW_PointOUR]->MoveDelta(delta);
      variables[BoxW_PointODR]->MoveDelta(delta);
      variables[BoxW_PointODL]->MoveDelta(delta);
      break;
   }
}

