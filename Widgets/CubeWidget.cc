
/*
 *  CubeWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Widgets/CubeWidget.h>
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

enum { CubeW_ConstIULODR, CubeW_ConstOULIDR, CubeW_ConstIDLOUR, CubeW_ConstODLIUR,
       CubeW_ConstHypo, CubeW_ConstDiag,
       CubeW_ConstIULUR, CubeW_ConstIULDL, CubeW_ConstIDRUR, CubeW_ConstIDRDL,
       CubeW_ConstMULUL, CubeW_ConstMURUR, CubeW_ConstMDRDR, CubeW_ConstMDLDL,
       CubeW_ConstOULUR, CubeW_ConstOULDL, CubeW_ConstODRUR, CubeW_ConstODRDL };
enum { CubeW_SphereIUL, CubeW_SphereIUR, CubeW_SphereIDR, CubeW_SphereIDL,
       CubeW_SphereOUL, CubeW_SphereOUR, CubeW_SphereODR, CubeW_SphereODL,
       CubeW_CylIU, CubeW_CylIR, CubeW_CylID, CubeW_CylIL,
       CubeW_CylMU, CubeW_CylMR, CubeW_CylMD, CubeW_CylML,
       CubeW_CylOU, CubeW_CylOR, CubeW_CylOD, CubeW_CylOL };
enum { CubeW_PointMatl, CubeW_EdgeMatl, CubeW_HighMatl };
enum { CubeW_PickSphIUL, CubeW_PickSphIUR, CubeW_PickSphIDR, CubeW_PickSphIDL,
       CubeW_PickSphOUL, CubeW_PickSphOUR, CubeW_PickSphODR, CubeW_PickSphODL,
       CubeW_PickCyls };

CubeWidget::CubeWidget( Module* module, double widget_scale )
: BaseWidget(module, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale)
{
   cerr << "Starting CubeWidget CTOR" << endl;
   Real INIT = 1.0*widget_scale;
   variables[CubeW_PointIUL] = new Variable("PntIUL", Scheme1, Point(0, 0, 0));
   variables[CubeW_PointIUR] = new Variable("PntIUR", Scheme2, Point(INIT, 0, 0));
   variables[CubeW_PointIDR] = new Variable("PntIDR", Scheme1, Point(INIT, INIT, 0));
   variables[CubeW_PointIDL] = new Variable("PntIDL", Scheme2, Point(0, INIT, 0));
   variables[CubeW_PointOUL] = new Variable("PntOUL", Scheme1, Point(0, 0, INIT));
   variables[CubeW_PointOUR] = new Variable("PntOUR", Scheme2, Point(INIT, 0, INIT));
   variables[CubeW_PointODR] = new Variable("PntODR", Scheme1, Point(INIT, INIT, INIT));
   variables[CubeW_PointODL] = new Variable("PntODL", Scheme2, Point(0, INIT, INIT));
   variables[CubeW_Dist] = new Variable("DIST", Scheme1, Point(INIT, 0, 0));
   variables[CubeW_Hypo] = new Variable("HYPO", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));
   variables[CubeW_Diag] = new Variable("DIAG", Scheme1, Point(sqrt(3*INIT*INIT), 0, 0));

   NOT_FINISHED("Constraints not right!");
   
   constraints[CubeW_ConstIULODR] = new DistanceConstraint("ConstIULODR",
							   NumSchemes,
							   variables[CubeW_PointIUL],
							   variables[CubeW_PointODR],
							   variables[CubeW_Diag]);
   constraints[CubeW_ConstIULODR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[CubeW_ConstIULODR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[CubeW_ConstIULODR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[CubeW_ConstIULODR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[CubeW_ConstIULODR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[CubeW_ConstOULIDR] = new DistanceConstraint("ConstOULIDR",
							   NumSchemes,
							   variables[CubeW_PointOUL],
							   variables[CubeW_PointIDR],
							   variables[CubeW_Diag]);
   constraints[CubeW_ConstOULIDR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[CubeW_ConstOULIDR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[CubeW_ConstOULIDR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[CubeW_ConstOULIDR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[CubeW_ConstOULIDR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[CubeW_ConstIDLOUR] = new DistanceConstraint("ConstIDLOUR",
							  NumSchemes,
							  variables[CubeW_PointIDL],
							  variables[CubeW_PointOUR],
							  variables[CubeW_Diag]);
   constraints[CubeW_ConstIDLOUR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[CubeW_ConstIDLOUR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[CubeW_ConstIDLOUR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[CubeW_ConstIDLOUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[CubeW_ConstIDLOUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[CubeW_ConstODLIUR] = new DistanceConstraint("ConstODLIUR",
							  NumSchemes,
							  variables[CubeW_PointODL],
							  variables[CubeW_PointIUR],
							  variables[CubeW_Diag]);
   constraints[CubeW_ConstODLIUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[CubeW_ConstODLIUR]->VarChoices(Scheme2, 2, 2, 1);
   constraints[CubeW_ConstODLIUR]->VarChoices(Scheme3, 1, 0, 1);
   constraints[CubeW_ConstODLIUR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[CubeW_ConstODLIUR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[CubeW_ConstHypo] = new HypotenuseConstraint("ConstHypo",
							   NumSchemes,
							   variables[CubeW_Dist],
							   variables[CubeW_Hypo]);
   constraints[CubeW_ConstHypo]->VarChoices(Scheme1, 1, 0);
   constraints[CubeW_ConstHypo]->VarChoices(Scheme2, 1, 0);
   constraints[CubeW_ConstHypo]->VarChoices(Scheme3, 1, 0);
   constraints[CubeW_ConstHypo]->VarChoices(Scheme4, 1, 0);
   constraints[CubeW_ConstHypo]->Priorities(P_Highest, P_Default);
   constraints[CubeW_ConstDiag] = new PythagorasConstraint("ConstDiag",
							   NumSchemes,
							   variables[CubeW_Dist],
							   variables[CubeW_Hypo],
							   variables[CubeW_Diag]);
   constraints[CubeW_ConstDiag]->VarChoices(Scheme1, 2, 2, 1);
   constraints[CubeW_ConstDiag]->VarChoices(Scheme2, 2, 2, 1);
   constraints[CubeW_ConstDiag]->VarChoices(Scheme3, 2, 2, 1);
   constraints[CubeW_ConstDiag]->VarChoices(Scheme4, 2, 2, 1);
   constraints[CubeW_ConstDiag]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[CubeW_ConstIULUR] = new DistanceConstraint("ConstIULUR",
							  NumSchemes,
							  variables[CubeW_PointIUL],
							  variables[CubeW_PointIUR],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstIULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstIULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstIULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstIULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstIULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstIULDL] = new DistanceConstraint("ConstIULDL",
							  NumSchemes,
							  variables[CubeW_PointIUL],
							  variables[CubeW_PointIDL],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstIULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstIULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstIULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstIULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstIULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstIDRUR] = new DistanceConstraint("ConstIDRUR",
							  NumSchemes,
							  variables[CubeW_PointIDR],
							  variables[CubeW_PointIUR],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstIDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstIDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstIDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstIDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstIDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstIDRDL] = new DistanceConstraint("ConstIDRUR",
							  NumSchemes,
							  variables[CubeW_PointIDR],
							  variables[CubeW_PointIDL],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstIDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstIDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstIDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstIDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstIDRDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstMULUL] = new DistanceConstraint("ConstMULUL",
							  NumSchemes,
							  variables[CubeW_PointIUL],
							  variables[CubeW_PointOUL],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstMULUL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstMULUL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstMULUL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstMULUL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstMULUL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstMURUR] = new DistanceConstraint("ConstMURUR",
							  NumSchemes,
							  variables[CubeW_PointIUR],
							  variables[CubeW_PointOUR],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstMURUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstMURUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstMURUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstMURUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstMURUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstMDRDR] = new DistanceConstraint("ConstMDRDR",
							  NumSchemes,
							  variables[CubeW_PointIDR],
							  variables[CubeW_PointODR],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstMDRDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstMDRDR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstMDRDR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstMDRDR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstMDRDR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstMDLDL] = new DistanceConstraint("ConstMDLDL",
							  NumSchemes,
							  variables[CubeW_PointIDL],
							  variables[CubeW_PointODL],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstMDLDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstMDLDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstMDLDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstMDLDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstMDLDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstOULUR] = new DistanceConstraint("ConstOULUR",
							  NumSchemes,
							  variables[CubeW_PointOUL],
							  variables[CubeW_PointOUR],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstOULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstOULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstOULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstOULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstOULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstOULDL] = new DistanceConstraint("ConstOULDL",
							  NumSchemes,
							  variables[CubeW_PointOUL],
							  variables[CubeW_PointODL],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstOULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstOULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstOULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstOULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstOULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstODRUR] = new DistanceConstraint("ConstODRUR",
							  NumSchemes,
							  variables[CubeW_PointODR],
							  variables[CubeW_PointOUR],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstODRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstODRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstODRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstODRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstODRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[CubeW_ConstODRDL] = new DistanceConstraint("ConstODRDL",
							  NumSchemes,
							  variables[CubeW_PointODR],
							  variables[CubeW_PointODL],
							  variables[CubeW_Dist]);
   constraints[CubeW_ConstODRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[CubeW_ConstODRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[CubeW_ConstODRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[CubeW_ConstODRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[CubeW_ConstODRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[CubeW_PointMatl] = new Material(Color(0,0,0), Color(.54, .60, 1),
						 Color(.5,.5,.5), 20);
   materials[CubeW_EdgeMatl] = new Material(Color(0,0,0), Color(.54, .60, .66),
						Color(.5,.5,.5), 20);
   materials[CubeW_HighMatl] = new Material(Color(0,0,0), Color(.7,.7,.7),
						Color(0,0,.6), 20);

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = CubeW_SphereIUL, pick = CubeW_PickSphIUL;
	geom <= CubeW_SphereODL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[CubeW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[CubeW_PointMatl]);
   
   GeomGroup* cyls = new GeomGroup;
   for (geom = CubeW_CylIU; geom <= CubeW_CylOL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[CubeW_PickCyls] = new GeomPick(cyls, module);
   picks[CubeW_PickCyls]->set_highlight(materials[CubeW_HighMatl]);
   picks[CubeW_PickCyls]->set_cbdata((void*)CubeW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[CubeW_PickCyls], materials[CubeW_EdgeMatl]);

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
   cerr << "Done with CubeWidget CTOR" << endl;
}


CubeWidget::~CubeWidget()
{
}


void
CubeWidget::execute()
{
   ((GeomSphere*)geometries[CubeW_SphereIUL])->move(variables[CubeW_PointIUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[CubeW_SphereIUR])->move(variables[CubeW_PointIUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[CubeW_SphereIDR])->move(variables[CubeW_PointIDR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[CubeW_SphereIDL])->move(variables[CubeW_PointIDL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[CubeW_SphereOUL])->move(variables[CubeW_PointOUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[CubeW_SphereOUR])->move(variables[CubeW_PointOUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[CubeW_SphereODR])->move(variables[CubeW_PointODR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[CubeW_SphereODL])->move(variables[CubeW_PointODL]->Get(),
						    1*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylIU])->move(variables[CubeW_PointIUL]->Get(),
						  variables[CubeW_PointIUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylIR])->move(variables[CubeW_PointIUR]->Get(),
						  variables[CubeW_PointIDR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylID])->move(variables[CubeW_PointIDR]->Get(),
						  variables[CubeW_PointIDL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylIL])->move(variables[CubeW_PointIDL]->Get(),
						  variables[CubeW_PointIUL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylMU])->move(variables[CubeW_PointIUL]->Get(),
						  variables[CubeW_PointOUL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylMR])->move(variables[CubeW_PointIUR]->Get(),
						  variables[CubeW_PointOUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylMD])->move(variables[CubeW_PointIDR]->Get(),
						  variables[CubeW_PointODR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylML])->move(variables[CubeW_PointIDL]->Get(),
						  variables[CubeW_PointODL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylOU])->move(variables[CubeW_PointOUL]->Get(),
						  variables[CubeW_PointOUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylOR])->move(variables[CubeW_PointOUR]->Get(),
						  variables[CubeW_PointODR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylOD])->move(variables[CubeW_PointODR]->Get(),
						  variables[CubeW_PointODL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[CubeW_CylOL])->move(variables[CubeW_PointODL]->Get(),
						  variables[CubeW_PointOUL]->Get(),
						  0.5*widget_scale);

   Vector spvec1(variables[CubeW_PointIUR]->Get() - variables[CubeW_PointIUL]->Get());
   Vector spvec2(variables[CubeW_PointIDL]->Get() - variables[CubeW_PointIUL]->Get());
   spvec1.normalize();
   spvec2.normalize();
   Vector v = Cross(spvec1, spvec2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      picks[geom]->set_principal(spvec1, spvec2, v);
   }
}

void
CubeWidget::geom_moved( int /* axis*/, double /*dist*/, const Vector& delta,
			void* cbdata )
{
   switch((int)cbdata){
   case CubeW_PickSphIUL:
      variables[CubeW_PointIUL]->SetDelta(delta);
      break;
   case CubeW_PickSphIUR:
      variables[CubeW_PointIUR]->SetDelta(delta);
      break;
   case CubeW_PickSphIDR:
      variables[CubeW_PointIDR]->SetDelta(delta);
      break;
   case CubeW_PickSphIDL:
      variables[CubeW_PointIDL]->SetDelta(delta);
      break;
   case CubeW_PickSphOUL:
      variables[CubeW_PointOUL]->SetDelta(delta);
      break;
   case CubeW_PickSphOUR:
      variables[CubeW_PointOUR]->SetDelta(delta);
      break;
   case CubeW_PickSphODR:
      variables[CubeW_PointODR]->SetDelta(delta);
      break;
   case CubeW_PickSphODL:
      variables[CubeW_PointODL]->SetDelta(delta);
      break;
   case CubeW_PickCyls:
      variables[CubeW_PointIUL]->MoveDelta(delta);
      variables[CubeW_PointIUR]->MoveDelta(delta);
      variables[CubeW_PointIDR]->MoveDelta(delta);
      variables[CubeW_PointIDL]->MoveDelta(delta);
      variables[CubeW_PointOUL]->MoveDelta(delta);
      variables[CubeW_PointOUR]->MoveDelta(delta);
      variables[CubeW_PointODR]->MoveDelta(delta);
      variables[CubeW_PointODL]->MoveDelta(delta);
      break;
   }
}

