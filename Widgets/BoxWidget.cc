
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
const Index NumMatls = 4;
const Index NumSchemes = 4;
const Index NumPcks = 15;

enum { ConstIULODR, ConstOULIDR, ConstIDLOUR, ConstODLIUR,
       ConstHypo, ConstDiag,
       ConstIULUR, ConstIULDL, ConstIDRUR, ConstIDRDL,
       ConstMULUL, ConstMURUR, ConstMDRDR, ConstMDLDL,
       ConstOULUR, ConstOULDL, ConstODRUR, ConstODRDL };
enum { SphereIUL, SphereIUR, SphereIDR, SphereIDL,
       SphereOUL, SphereOUR, SphereODR, SphereODL,
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
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale)
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

   materials[PointMatl] = PointWidgetMaterial;
   materials[EdgeMatl] = EdgeWidgetMaterial;
   materials[ResizeMatl] = ResizeWidgetMaterial;
   materials[HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = SphereIUL, pick = PickSphIUL;
	geom <= SphereODL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[PointMatl]);
   
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
      picks[pick]->set_highlight(materials[HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      resizes->add(picks[pick]);
   }
   GeomMaterial* resizem = new GeomMaterial(resizes, materials[ResizeMatl]);

   GeomGroup* cyls = new GeomGroup;
   for (geom = CylIU; geom <= CylOL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[PickCyls] = new GeomPick(cyls, module);
   picks[PickCyls]->set_highlight(materials[HighMatl]);
   picks[PickCyls]->set_cbdata((void*)PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[PickCyls], materials[EdgeMatl]);

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

   SetEpsilon(widget_scale*1e-6);

   Vector spvec1(variables[PointIURVar]->point() - variables[PointIULVar]->point());
   Vector spvec2(variables[PointIDLVar]->point() - variables[PointIULVar]->point());
   Vector spvec3(variables[PointOULVar]->point() - variables[PointIULVar]->point());
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
   case PickCyls:
      MoveDelta(delta);
      break;
   }
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
}


Point
BoxWidget::ReferencePoint() const
{
   return (variables[PointIULVar]->point()
	   + (variables[PointODRVar]->point()
	      -variables[PointIULVar]->point())/2.0);
}
