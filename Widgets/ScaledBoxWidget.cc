
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
#include <Constraints/HypotenuseConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 18;
const Index NumVars = 11;
const Index NumGeoms = 20;
const Index NumMatls = 4;
const Index NumSchemes = 4;
const Index NumPcks = 9;

enum { SBoxW_ConstIULODR, SBoxW_ConstOULIDR, SBoxW_ConstIDLOUR, SBoxW_ConstODLIUR,
       SBoxW_ConstHypo, SBoxW_ConstDiag,
       SBoxW_ConstIULUR, SBoxW_ConstIULDL, SBoxW_ConstIDRUR, SBoxW_ConstIDRDL,
       SBoxW_ConstMULUL, SBoxW_ConstMURUR, SBoxW_ConstMDRDR, SBoxW_ConstMDLDL,
       SBoxW_ConstOULUR, SBoxW_ConstOULDL, SBoxW_ConstODRUR, SBoxW_ConstODRDL };
enum { SBoxW_SphereIUL, SBoxW_SphereIUR, SBoxW_SphereIDR, SBoxW_SphereIDL,
       SBoxW_SphereOUL, SBoxW_SphereOUR, SBoxW_SphereODR, SBoxW_SphereODL,
       SBoxW_CylIU, SBoxW_CylIR, SBoxW_CylID, SBoxW_CylIL,
       SBoxW_CylMU, SBoxW_CylMR, SBoxW_CylMD, SBoxW_CylML,
       SBoxW_CylOU, SBoxW_CylOR, SBoxW_CylOD, SBoxW_CylOL };
enum { SBoxW_PickSphIUL, SBoxW_PickSphIUR, SBoxW_PickSphIDR, SBoxW_PickSphIDL,
       SBoxW_PickSphOUL, SBoxW_PickSphOUR, SBoxW_PickSphODR, SBoxW_PickSphODL,
       SBoxW_PickCyls };

ScaledBoxWidget::ScaledBoxWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale)
{
   Real INIT = 1.0*widget_scale;
   variables[SBoxW_PointIUL] = new Variable("PntIUL", Scheme1, Point(0, 0, 0));
   variables[SBoxW_PointIUR] = new Variable("PntIUR", Scheme2, Point(INIT, 0, 0));
   variables[SBoxW_PointIDR] = new Variable("PntIDR", Scheme1, Point(INIT, INIT, 0));
   variables[SBoxW_PointIDL] = new Variable("PntIDL", Scheme2, Point(0, INIT, 0));
   variables[SBoxW_PointOUL] = new Variable("PntOUL", Scheme1, Point(0, 0, INIT));
   variables[SBoxW_PointOUR] = new Variable("PntOUR", Scheme2, Point(INIT, 0, INIT));
   variables[SBoxW_PointODR] = new Variable("PntODR", Scheme1, Point(INIT, INIT, INIT));
   variables[SBoxW_PointODL] = new Variable("PntODL", Scheme2, Point(0, INIT, INIT));
   variables[SBoxW_Dist] = new Variable("DIST", Scheme1, Point(INIT, 0, 0));
   variables[SBoxW_Hypo] = new Variable("HYPO", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));
   variables[SBoxW_Diag] = new Variable("DIAG", Scheme1, Point(sqrt(3*INIT*INIT), 0, 0));

   NOT_FINISHED("Constraints not right!");
   
   constraints[SBoxW_ConstIULODR] = new DistanceConstraint("ConstIULODR",
							   NumSchemes,
							   variables[SBoxW_PointIUL],
							   variables[SBoxW_PointODR],
							   variables[SBoxW_Diag]);
   constraints[SBoxW_ConstIULODR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[SBoxW_ConstIULODR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SBoxW_ConstIULODR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[SBoxW_ConstIULODR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SBoxW_ConstIULODR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SBoxW_ConstOULIDR] = new DistanceConstraint("ConstOULIDR",
							   NumSchemes,
							   variables[SBoxW_PointOUL],
							   variables[SBoxW_PointIDR],
							   variables[SBoxW_Diag]);
   constraints[SBoxW_ConstOULIDR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SBoxW_ConstOULIDR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SBoxW_ConstOULIDR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SBoxW_ConstOULIDR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[SBoxW_ConstOULIDR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SBoxW_ConstIDLOUR] = new DistanceConstraint("ConstIDLOUR",
							  NumSchemes,
							  variables[SBoxW_PointIDL],
							  variables[SBoxW_PointOUR],
							  variables[SBoxW_Diag]);
   constraints[SBoxW_ConstIDLOUR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[SBoxW_ConstIDLOUR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SBoxW_ConstIDLOUR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[SBoxW_ConstIDLOUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SBoxW_ConstIDLOUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SBoxW_ConstODLIUR] = new DistanceConstraint("ConstODLIUR",
							  NumSchemes,
							  variables[SBoxW_PointODL],
							  variables[SBoxW_PointIUR],
							  variables[SBoxW_Diag]);
   constraints[SBoxW_ConstODLIUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SBoxW_ConstODLIUR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SBoxW_ConstODLIUR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SBoxW_ConstODLIUR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[SBoxW_ConstODLIUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SBoxW_ConstHypo] = new HypotenuseConstraint("ConstHypo",
							   NumSchemes,
							   variables[SBoxW_Dist],
							   variables[SBoxW_Hypo]);
   constraints[SBoxW_ConstHypo]->VarChoices(Scheme1, 1, 0);
   constraints[SBoxW_ConstHypo]->VarChoices(Scheme2, 1, 0);
   constraints[SBoxW_ConstHypo]->VarChoices(Scheme3, 1, 0);
   constraints[SBoxW_ConstHypo]->VarChoices(Scheme4, 1, 0);
   constraints[SBoxW_ConstHypo]->Priorities(P_Highest, P_Default);
   constraints[SBoxW_ConstDiag] = new PythagorasConstraint("ConstDiag",
							   NumSchemes,
							   variables[SBoxW_Dist],
							   variables[SBoxW_Hypo],
							   variables[SBoxW_Diag]);
   constraints[SBoxW_ConstDiag]->VarChoices(Scheme1, 2, 2, 1);
   constraints[SBoxW_ConstDiag]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SBoxW_ConstDiag]->VarChoices(Scheme3, 2, 2, 1);
   constraints[SBoxW_ConstDiag]->VarChoices(Scheme4, 2, 2, 1);
   constraints[SBoxW_ConstDiag]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SBoxW_ConstIULUR] = new DistanceConstraint("ConstIULUR",
							  NumSchemes,
							  variables[SBoxW_PointIUL],
							  variables[SBoxW_PointIUR],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstIULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstIULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstIULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstIULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstIULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstIULDL] = new DistanceConstraint("ConstIULDL",
							  NumSchemes,
							  variables[SBoxW_PointIUL],
							  variables[SBoxW_PointIDL],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstIULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstIULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstIULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstIULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstIULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstIDRUR] = new DistanceConstraint("ConstIDRUR",
							  NumSchemes,
							  variables[SBoxW_PointIDR],
							  variables[SBoxW_PointIUR],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstIDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstIDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstIDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstIDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstIDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstIDRDL] = new DistanceConstraint("ConstIDRUR",
							  NumSchemes,
							  variables[SBoxW_PointIDR],
							  variables[SBoxW_PointIDL],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstIDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstIDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstIDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstIDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstIDRDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstMULUL] = new DistanceConstraint("ConstMULUL",
							  NumSchemes,
							  variables[SBoxW_PointIUL],
							  variables[SBoxW_PointOUL],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstMULUL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstMULUL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstMULUL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstMULUL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstMULUL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstMURUR] = new DistanceConstraint("ConstMURUR",
							  NumSchemes,
							  variables[SBoxW_PointIUR],
							  variables[SBoxW_PointOUR],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstMURUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstMURUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstMURUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstMURUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstMURUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstMDRDR] = new DistanceConstraint("ConstMDRDR",
							  NumSchemes,
							  variables[SBoxW_PointIDR],
							  variables[SBoxW_PointODR],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstMDRDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstMDRDR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstMDRDR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstMDRDR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstMDRDR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstMDLDL] = new DistanceConstraint("ConstMDLDL",
							  NumSchemes,
							  variables[SBoxW_PointIDL],
							  variables[SBoxW_PointODL],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstMDLDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstMDLDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstMDLDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstMDLDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstMDLDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstOULUR] = new DistanceConstraint("ConstOULUR",
							  NumSchemes,
							  variables[SBoxW_PointOUL],
							  variables[SBoxW_PointOUR],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstOULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstOULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstOULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstOULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstOULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstOULDL] = new DistanceConstraint("ConstOULDL",
							  NumSchemes,
							  variables[SBoxW_PointOUL],
							  variables[SBoxW_PointODL],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstOULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstOULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstOULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstOULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstOULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstODRUR] = new DistanceConstraint("ConstODRUR",
							  NumSchemes,
							  variables[SBoxW_PointODR],
							  variables[SBoxW_PointOUR],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstODRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstODRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstODRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstODRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstODRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SBoxW_ConstODRDL] = new DistanceConstraint("ConstODRDL",
							  NumSchemes,
							  variables[SBoxW_PointODR],
							  variables[SBoxW_PointODL],
							  variables[SBoxW_Dist]);
   constraints[SBoxW_ConstODRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SBoxW_ConstODRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SBoxW_ConstODRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SBoxW_ConstODRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SBoxW_ConstODRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[SBoxW_PointMatl] = PointWidgetMaterial;
   materials[SBoxW_EdgeMatl] = EdgeWidgetMaterial;
   materials[SBoxW_SliderMatl] = SliderWidgetMaterial;
   materials[SBoxW_HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = SBoxW_SphereIUL, pick = SBoxW_PickSphIUL;
	geom <= SBoxW_SphereODL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[SBoxW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[SBoxW_PointMatl]);
   
   GeomGroup* cyls = new GeomGroup;
   for (geom = SBoxW_CylIU; geom <= SBoxW_CylOL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[SBoxW_PickCyls] = new GeomPick(cyls, module);
   picks[SBoxW_PickCyls]->set_highlight(materials[SBoxW_HighMatl]);
   picks[SBoxW_PickCyls]->set_cbdata((void*)SBoxW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[SBoxW_PickCyls], materials[SBoxW_EdgeMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(cylsm);

   SetEpsilon(widget_scale*1e-4);

   FinishWidget(w);
}


ScaledBoxWidget::~ScaledBoxWidget()
{
}


void
ScaledBoxWidget::widget_execute()
{
   ((GeomSphere*)geometries[SBoxW_SphereIUL])->move(variables[SBoxW_PointIUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SBoxW_SphereIUR])->move(variables[SBoxW_PointIUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SBoxW_SphereIDR])->move(variables[SBoxW_PointIDR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SBoxW_SphereIDL])->move(variables[SBoxW_PointIDL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SBoxW_SphereOUL])->move(variables[SBoxW_PointOUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SBoxW_SphereOUR])->move(variables[SBoxW_PointOUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SBoxW_SphereODR])->move(variables[SBoxW_PointODR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SBoxW_SphereODL])->move(variables[SBoxW_PointODL]->Get(),
						    1*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylIU])->move(variables[SBoxW_PointIUL]->Get(),
						  variables[SBoxW_PointIUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylIR])->move(variables[SBoxW_PointIUR]->Get(),
						  variables[SBoxW_PointIDR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylID])->move(variables[SBoxW_PointIDR]->Get(),
						  variables[SBoxW_PointIDL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylIL])->move(variables[SBoxW_PointIDL]->Get(),
						  variables[SBoxW_PointIUL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylMU])->move(variables[SBoxW_PointIUL]->Get(),
						  variables[SBoxW_PointOUL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylMR])->move(variables[SBoxW_PointIUR]->Get(),
						  variables[SBoxW_PointOUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylMD])->move(variables[SBoxW_PointIDR]->Get(),
						  variables[SBoxW_PointODR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylML])->move(variables[SBoxW_PointIDL]->Get(),
						  variables[SBoxW_PointODL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylOU])->move(variables[SBoxW_PointOUL]->Get(),
						  variables[SBoxW_PointOUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylOR])->move(variables[SBoxW_PointOUR]->Get(),
						  variables[SBoxW_PointODR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylOD])->move(variables[SBoxW_PointODR]->Get(),
						  variables[SBoxW_PointODL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SBoxW_CylOL])->move(variables[SBoxW_PointODL]->Get(),
						  variables[SBoxW_PointOUL]->Get(),
						  0.5*widget_scale);

   Vector spvec1(variables[SBoxW_PointIUR]->Get() - variables[SBoxW_PointIUL]->Get());
   Vector spvec2(variables[SBoxW_PointIDL]->Get() - variables[SBoxW_PointIUL]->Get());
   Vector spvec3(variables[SBoxW_PointOUL]->Get() - variables[SBoxW_PointIUL]->Get());
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
ScaledBoxWidget::geom_moved( int /* axis*/, double /*dist*/, const Vector& delta,
			void* cbdata )
{
   switch((int)cbdata){
   case SBoxW_PickSphIUL:
      variables[SBoxW_PointIUL]->SetDelta(delta);
      break;
   case SBoxW_PickSphIUR:
      variables[SBoxW_PointIUR]->SetDelta(delta);
      break;
   case SBoxW_PickSphIDR:
      variables[SBoxW_PointIDR]->SetDelta(delta);
      break;
   case SBoxW_PickSphIDL:
      variables[SBoxW_PointIDL]->SetDelta(delta);
      break;
   case SBoxW_PickSphOUL:
      variables[SBoxW_PointOUL]->SetDelta(delta);
      break;
   case SBoxW_PickSphOUR:
      variables[SBoxW_PointOUR]->SetDelta(delta);
      break;
   case SBoxW_PickSphODR:
      variables[SBoxW_PointODR]->SetDelta(delta);
      break;
   case SBoxW_PickSphODL:
      variables[SBoxW_PointODL]->SetDelta(delta);
      break;
   case SBoxW_PickCyls:
      variables[SBoxW_PointIUL]->MoveDelta(delta);
      variables[SBoxW_PointIUR]->MoveDelta(delta);
      variables[SBoxW_PointIDR]->MoveDelta(delta);
      variables[SBoxW_PointIDL]->MoveDelta(delta);
      variables[SBoxW_PointOUL]->MoveDelta(delta);
      variables[SBoxW_PointOUR]->MoveDelta(delta);
      variables[SBoxW_PointODR]->MoveDelta(delta);
      variables[SBoxW_PointODL]->MoveDelta(delta);
      break;
   }
}

