
/*
 *  ScaledCubeWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/ScaledCubeWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/HypotenuseConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 18;
const Index NumVars = 11;
const Index NumGeoms = 20;
const Index NumMatls = 3;
const Index NumSchemes = 4;
const Index NumPcks = 9;

enum { SCubeW_ConstIULODR, SCubeW_ConstOULIDR, SCubeW_ConstIDLOUR, SCubeW_ConstODLIUR,
       SCubeW_ConstHypo, SCubeW_ConstDiag,
       SCubeW_ConstIULUR, SCubeW_ConstIULDL, SCubeW_ConstIDRUR, SCubeW_ConstIDRDL,
       SCubeW_ConstMULUL, SCubeW_ConstMURUR, SCubeW_ConstMDRDR, SCubeW_ConstMDLDL,
       SCubeW_ConstOULUR, SCubeW_ConstOULDL, SCubeW_ConstODRUR, SCubeW_ConstODRDL };
enum { SCubeW_SphereIUL, SCubeW_SphereIUR, SCubeW_SphereIDR, SCubeW_SphereIDL,
       SCubeW_SphereOUL, SCubeW_SphereOUR, SCubeW_SphereODR, SCubeW_SphereODL,
       SCubeW_CylIU, SCubeW_CylIR, SCubeW_CylID, SCubeW_CylIL,
       SCubeW_CylMU, SCubeW_CylMR, SCubeW_CylMD, SCubeW_CylML,
       SCubeW_CylOU, SCubeW_CylOR, SCubeW_CylOD, SCubeW_CylOL };
enum { SCubeW_PointMatl, SCubeW_EdgeMatl, SCubeW_HighMatl };
enum { SCubeW_PickSphIUL, SCubeW_PickSphIUR, SCubeW_PickSphIDR, SCubeW_PickSphIDL,
       SCubeW_PickSphOUL, SCubeW_PickSphOUR, SCubeW_PickSphODR, SCubeW_PickSphODL,
       SCubeW_PickCyls };

ScaledCubeWidget::ScaledCubeWidget( Module* module, double widget_scale )
: BaseWidget(module, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale)
{
   cerr << "Starting ScaledCubeWidget CTOR" << endl;
   Real INIT = 1.0*widget_scale;
   variables[SCubeW_PointIUL] = new Variable("PntIUL", Scheme1, Point(0, 0, 0));
   variables[SCubeW_PointIUR] = new Variable("PntIUR", Scheme2, Point(INIT, 0, 0));
   variables[SCubeW_PointIDR] = new Variable("PntIDR", Scheme1, Point(INIT, INIT, 0));
   variables[SCubeW_PointIDL] = new Variable("PntIDL", Scheme2, Point(0, INIT, 0));
   variables[SCubeW_PointOUL] = new Variable("PntOUL", Scheme1, Point(0, 0, INIT));
   variables[SCubeW_PointOUR] = new Variable("PntOUR", Scheme2, Point(INIT, 0, INIT));
   variables[SCubeW_PointODR] = new Variable("PntODR", Scheme1, Point(INIT, INIT, INIT));
   variables[SCubeW_PointODL] = new Variable("PntODL", Scheme2, Point(0, INIT, INIT));
   variables[SCubeW_Dist] = new Variable("DIST", Scheme1, Point(INIT, 0, 0));
   variables[SCubeW_Hypo] = new Variable("HYPO", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));
   variables[SCubeW_Diag] = new Variable("DIAG", Scheme1, Point(sqrt(3*INIT*INIT), 0, 0));

   NOT_FINISHED("Constraints not right!");
   
   constraints[SCubeW_ConstIULODR] = new DistanceConstraint("ConstIULODR",
							   NumSchemes,
							   variables[SCubeW_PointIUL],
							   variables[SCubeW_PointODR],
							   variables[SCubeW_Diag]);
   constraints[SCubeW_ConstIULODR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[SCubeW_ConstIULODR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SCubeW_ConstIULODR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[SCubeW_ConstIULODR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SCubeW_ConstIULODR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SCubeW_ConstOULIDR] = new DistanceConstraint("ConstOULIDR",
							   NumSchemes,
							   variables[SCubeW_PointOUL],
							   variables[SCubeW_PointIDR],
							   variables[SCubeW_Diag]);
   constraints[SCubeW_ConstOULIDR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SCubeW_ConstOULIDR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SCubeW_ConstOULIDR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SCubeW_ConstOULIDR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[SCubeW_ConstOULIDR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SCubeW_ConstIDLOUR] = new DistanceConstraint("ConstIDLOUR",
							  NumSchemes,
							  variables[SCubeW_PointIDL],
							  variables[SCubeW_PointOUR],
							  variables[SCubeW_Diag]);
   constraints[SCubeW_ConstIDLOUR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[SCubeW_ConstIDLOUR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[SCubeW_ConstIDLOUR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[SCubeW_ConstIDLOUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[SCubeW_ConstIDLOUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SCubeW_ConstODLIUR] = new DistanceConstraint("ConstODLIUR",
							  NumSchemes,
							  variables[SCubeW_PointODL],
							  variables[SCubeW_PointIUR],
							  variables[SCubeW_Diag]);
   constraints[SCubeW_ConstODLIUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[SCubeW_ConstODLIUR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SCubeW_ConstODLIUR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[SCubeW_ConstODLIUR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[SCubeW_ConstODLIUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SCubeW_ConstHypo] = new HypotenuseConstraint("ConstHypo",
							   NumSchemes,
							   variables[SCubeW_Dist],
							   variables[SCubeW_Hypo]);
   constraints[SCubeW_ConstHypo]->VarChoices(Scheme1, 1, 0);
   constraints[SCubeW_ConstHypo]->VarChoices(Scheme2, 1, 0);
   constraints[SCubeW_ConstHypo]->VarChoices(Scheme3, 1, 0);
   constraints[SCubeW_ConstHypo]->VarChoices(Scheme4, 1, 0);
   constraints[SCubeW_ConstHypo]->Priorities(P_Highest, P_Default);
   constraints[SCubeW_ConstDiag] = new PythagorasConstraint("ConstDiag",
							   NumSchemes,
							   variables[SCubeW_Dist],
							   variables[SCubeW_Hypo],
							   variables[SCubeW_Diag]);
   constraints[SCubeW_ConstDiag]->VarChoices(Scheme1, 2, 2, 1);
   constraints[SCubeW_ConstDiag]->VarChoices(Scheme2, 2, 2, 1);
   constraints[SCubeW_ConstDiag]->VarChoices(Scheme3, 2, 2, 1);
   constraints[SCubeW_ConstDiag]->VarChoices(Scheme4, 2, 2, 1);
   constraints[SCubeW_ConstDiag]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[SCubeW_ConstIULUR] = new DistanceConstraint("ConstIULUR",
							  NumSchemes,
							  variables[SCubeW_PointIUL],
							  variables[SCubeW_PointIUR],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstIULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstIULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstIULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstIULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstIULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstIULDL] = new DistanceConstraint("ConstIULDL",
							  NumSchemes,
							  variables[SCubeW_PointIUL],
							  variables[SCubeW_PointIDL],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstIULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstIULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstIULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstIULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstIULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstIDRUR] = new DistanceConstraint("ConstIDRUR",
							  NumSchemes,
							  variables[SCubeW_PointIDR],
							  variables[SCubeW_PointIUR],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstIDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstIDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstIDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstIDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstIDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstIDRDL] = new DistanceConstraint("ConstIDRUR",
							  NumSchemes,
							  variables[SCubeW_PointIDR],
							  variables[SCubeW_PointIDL],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstIDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstIDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstIDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstIDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstIDRDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstMULUL] = new DistanceConstraint("ConstMULUL",
							  NumSchemes,
							  variables[SCubeW_PointIUL],
							  variables[SCubeW_PointOUL],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstMULUL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstMULUL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstMULUL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstMULUL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstMULUL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstMURUR] = new DistanceConstraint("ConstMURUR",
							  NumSchemes,
							  variables[SCubeW_PointIUR],
							  variables[SCubeW_PointOUR],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstMURUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstMURUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstMURUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstMURUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstMURUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstMDRDR] = new DistanceConstraint("ConstMDRDR",
							  NumSchemes,
							  variables[SCubeW_PointIDR],
							  variables[SCubeW_PointODR],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstMDRDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstMDRDR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstMDRDR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstMDRDR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstMDRDR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstMDLDL] = new DistanceConstraint("ConstMDLDL",
							  NumSchemes,
							  variables[SCubeW_PointIDL],
							  variables[SCubeW_PointODL],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstMDLDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstMDLDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstMDLDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstMDLDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstMDLDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstOULUR] = new DistanceConstraint("ConstOULUR",
							  NumSchemes,
							  variables[SCubeW_PointOUL],
							  variables[SCubeW_PointOUR],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstOULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstOULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstOULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstOULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstOULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstOULDL] = new DistanceConstraint("ConstOULDL",
							  NumSchemes,
							  variables[SCubeW_PointOUL],
							  variables[SCubeW_PointODL],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstOULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstOULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstOULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstOULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstOULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstODRUR] = new DistanceConstraint("ConstODRUR",
							  NumSchemes,
							  variables[SCubeW_PointODR],
							  variables[SCubeW_PointOUR],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstODRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstODRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstODRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstODRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstODRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[SCubeW_ConstODRDL] = new DistanceConstraint("ConstODRDL",
							  NumSchemes,
							  variables[SCubeW_PointODR],
							  variables[SCubeW_PointODL],
							  variables[SCubeW_Dist]);
   constraints[SCubeW_ConstODRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[SCubeW_ConstODRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[SCubeW_ConstODRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[SCubeW_ConstODRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[SCubeW_ConstODRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[SCubeW_PointMatl] = new Material(Color(0,0,0), Color(.54, .60, 1),
						 Color(.5,.5,.5), 20);
   materials[SCubeW_EdgeMatl] = new Material(Color(0,0,0), Color(.54, .60, .66),
						Color(.5,.5,.5), 20);
   materials[SCubeW_HighMatl] = new Material(Color(0,0,0), Color(.7,.7,.7),
						Color(0,0,.6), 20);

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = SCubeW_SphereIUL, pick = SCubeW_PickSphIUL;
	geom <= SCubeW_SphereODL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[SCubeW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[SCubeW_PointMatl]);
   
   GeomGroup* cyls = new GeomGroup;
   for (geom = SCubeW_CylIU; geom <= SCubeW_CylOL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[SCubeW_PickCyls] = new GeomPick(cyls, module);
   picks[SCubeW_PickCyls]->set_highlight(materials[SCubeW_HighMatl]);
   picks[SCubeW_PickCyls]->set_cbdata((void*)SCubeW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[SCubeW_PickCyls], materials[SCubeW_EdgeMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(cylsm);

   FinishWidget(w);

   SetEpsilon(widget_scale*1e-4);

   // Init variables.
   for (Index vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Order();
   
   for (vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Resolve();
   cerr << "Done with ScaledCubeWidget CTOR" << endl;
}


ScaledCubeWidget::~ScaledCubeWidget()
{
}


void
ScaledCubeWidget::execute()
{
   ((GeomSphere*)geometries[SCubeW_SphereIUL])->move(variables[SCubeW_PointIUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SCubeW_SphereIUR])->move(variables[SCubeW_PointIUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SCubeW_SphereIDR])->move(variables[SCubeW_PointIDR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SCubeW_SphereIDL])->move(variables[SCubeW_PointIDL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SCubeW_SphereOUL])->move(variables[SCubeW_PointOUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SCubeW_SphereOUR])->move(variables[SCubeW_PointOUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SCubeW_SphereODR])->move(variables[SCubeW_PointODR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[SCubeW_SphereODL])->move(variables[SCubeW_PointODL]->Get(),
						    1*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylIU])->move(variables[SCubeW_PointIUL]->Get(),
						  variables[SCubeW_PointIUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylIR])->move(variables[SCubeW_PointIUR]->Get(),
						  variables[SCubeW_PointIDR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylID])->move(variables[SCubeW_PointIDR]->Get(),
						  variables[SCubeW_PointIDL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylIL])->move(variables[SCubeW_PointIDL]->Get(),
						  variables[SCubeW_PointIUL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylMU])->move(variables[SCubeW_PointIUL]->Get(),
						  variables[SCubeW_PointOUL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylMR])->move(variables[SCubeW_PointIUR]->Get(),
						  variables[SCubeW_PointOUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylMD])->move(variables[SCubeW_PointIDR]->Get(),
						  variables[SCubeW_PointODR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylML])->move(variables[SCubeW_PointIDL]->Get(),
						  variables[SCubeW_PointODL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylOU])->move(variables[SCubeW_PointOUL]->Get(),
						  variables[SCubeW_PointOUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylOR])->move(variables[SCubeW_PointOUR]->Get(),
						  variables[SCubeW_PointODR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylOD])->move(variables[SCubeW_PointODR]->Get(),
						  variables[SCubeW_PointODL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[SCubeW_CylOL])->move(variables[SCubeW_PointODL]->Get(),
						  variables[SCubeW_PointOUL]->Get(),
						  0.5*widget_scale);

   Vector spvec1(variables[SCubeW_PointIUR]->Get() - variables[SCubeW_PointIUL]->Get());
   Vector spvec2(variables[SCubeW_PointIDL]->Get() - variables[SCubeW_PointIUL]->Get());
   spvec1.normalize();
   spvec2.normalize();
   Vector v = Cross(spvec1, spvec2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      picks[geom]->set_principal(spvec1, spvec2, v);
   }
}

void
ScaledCubeWidget::geom_moved( int /* axis*/, double /*dist*/, const Vector& delta,
			void* cbdata )
{
   cerr << "Moved called..." << endl;
   switch((int)cbdata){
   case SCubeW_PickSphIUL:
      variables[SCubeW_PointIUL]->SetDelta(delta);
      break;
   case SCubeW_PickSphIUR:
      variables[SCubeW_PointIUR]->SetDelta(delta);
      break;
   case SCubeW_PickSphIDR:
      variables[SCubeW_PointIDR]->SetDelta(delta);
      break;
   case SCubeW_PickSphIDL:
      variables[SCubeW_PointIDL]->SetDelta(delta);
      break;
   case SCubeW_PickSphOUL:
      variables[SCubeW_PointOUL]->SetDelta(delta);
      break;
   case SCubeW_PickSphOUR:
      variables[SCubeW_PointOUR]->SetDelta(delta);
      break;
   case SCubeW_PickSphODR:
      variables[SCubeW_PointODR]->SetDelta(delta);
      break;
   case SCubeW_PickSphODL:
      variables[SCubeW_PointODL]->SetDelta(delta);
      break;
   case SCubeW_PickCyls:
      variables[SCubeW_PointIUL]->MoveDelta(delta);
      variables[SCubeW_PointIUR]->MoveDelta(delta);
      variables[SCubeW_PointIDR]->MoveDelta(delta);
      variables[SCubeW_PointIDL]->MoveDelta(delta);
      variables[SCubeW_PointOUL]->MoveDelta(delta);
      variables[SCubeW_PointOUR]->MoveDelta(delta);
      variables[SCubeW_PointODR]->MoveDelta(delta);
      variables[SCubeW_PointODL]->MoveDelta(delta);
      break;
   }
}

