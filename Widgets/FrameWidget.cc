
/*
 *  FrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include "FrameWidget.h"
#include <Constraints/DistanceConstraint.h>
#include <Constraints/HypotenousConstraint.h>


const Index NumCons = 7;
const Index NumVars = 6;
const Index NumGeoms = 8;
const Index NumMatls = 3;
const Index NumSchemes = 2;

enum { FrameW_ConstULDR, FrameW_ConstURDL, FrameW_ConstHypo,
       FrameW_ConstULUR, FrameW_ConstULDL, FrameW_ConstDRUR, FrameW_ConstDRDL };
enum { FrameW_SphereUL, FrameW_SphereUR, FrameW_SphereDR, FrameW_SphereDL,
       FrameW_CylU, FrameW_CylR, FrameW_CylD, FrameW_CylL };
enum { FrameW_PointMatl, FrameW_EdgeMatl, FrameW_HighMatl };

FrameWidget::FrameWidget( Module* module )
: BaseWidget(module, NumVars, NumCons, NumGeoms, NumMatls)
{
   variables[FrameW_PointUL] = new Variable("PntUL", Scheme1, Point(100,100,0));
   variables[FrameW_PointUR] = new Variable("PntUR", Scheme2, Point(200,100,0));
   variables[FrameW_PointDR] = new Variable("PntDR", Scheme1, Point(200,200,0));
   variables[FrameW_PointDL] = new Variable("PntDL", Scheme2, Point(100,200,0));
   variables[FrameW_Dist] = new Variable("DIST", Scheme1, Point(100,100,100));
   variables[FrameW_Hypo] = new Variable("HYPO", Scheme1, Point(sqrt(2*100*100),
								sqrt(2*100*100),
								sqrt(2*100*100)));
   
   constraints[FrameW_ConstULDR] = new DistanceConstraint("Const13",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointDR],
							  variables[FrameW_Hypo]);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme1, 2, 2, 1);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[FrameW_ConstULDR]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[FrameW_ConstURDL] = new DistanceConstraint("Const24",
							  NumSchemes,
							  variables[FrameW_PointUR],
							  variables[FrameW_PointDL],
							  variables[FrameW_Hypo]);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme2, 2, 2, 1);
   constraints[FrameW_ConstURDL]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[FrameW_ConstHypo] = new HypotenousConstraint("ConstHypo",
							    NumSchemes,
							    variables[FrameW_Hypo],
							    variables[FrameW_Dist]);
   constraints[FrameW_ConstHypo]->VarChoices(Scheme1, 1, 0);
   constraints[FrameW_ConstHypo]->VarChoices(Scheme2, 1, 0);
   constraints[FrameW_ConstHypo]->Priorities(P_Highest, P_Default);
   constraints[FrameW_ConstULUR] = new DistanceConstraint("Const12",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointUR],
							  variables[FrameW_Dist]);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstULDL] = new DistanceConstraint("Const14",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointDL],
							  variables[4]);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstDRUR] = new DistanceConstraint("Const32",
							  NumSchemes,
							  variables[FrameW_PointDR],
							  variables[FrameW_PointUR],
							  variables[FrameW_Dist]);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstDRDL] = new DistanceConstraint("Const34",
							  NumSchemes,
							  variables[FrameW_PointDR],
							  variables[FrameW_PointDL],
							  variables[FrameW_Dist]);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[FrameW_PointMatl] = new MaterialProp(Color(0,0,0), Color(.54, .60, 1),
						  Color(.5,.5,.5), 20);
   materials[FrameW_EdgeMatl] = new MaterialProp(Color(0,0,0), Color(.54, .60, .66),
						 Color(.5,.5,.5), 20);
   materials[FrameW_HighMatl] = new MaterialProp(Color(0,0,0), Color(.7,.7,.7),
						 Color(0,0,.6), 20);

   geometries[FrameW_SphereUL] = new GeomSphere;
   geometries[FrameW_SphereUL]->pick = new GeomPick(module);
   geometries[FrameW_SphereUL]->pick->set_highlight(materials[FrameW_HighMatl]);
   geometries[FrameW_SphereUL]->pick->set_cbdata((void*)FrameW_SphereUL);
   geometries[FrameW_SphereUL]->set_matl(materials[FrameW_PointMatl]);
   geometries[FrameW_SphereUR] = new GeomSphere;
   geometries[FrameW_SphereUR]->pick = new GeomPick(module);
   geometries[FrameW_SphereUR]->pick->set_highlight(materials[FrameW_HighMatl]);
   geometries[FrameW_SphereUR]->pick->set_cbdata((void*)FrameW_SphereUR);
   geometries[FrameW_SphereUR]->set_matl(materials[FrameW_PointMatl]);
   geometries[FrameW_SphereDR] = new GeomSphere;
   geometries[FrameW_SphereDR]->pick = new GeomPick(module);
   geometries[FrameW_SphereDR]->pick->set_highlight(materials[FrameW_HighMatl]);
   geometries[FrameW_SphereDR]->pick->set_cbdata((void*)FrameW_SphereDR);
   geometries[FrameW_SphereDR]->set_matl(materials[FrameW_PointMatl]);
   geometries[FrameW_SphereDL] = new GeomSphere;
   geometries[FrameW_SphereDL]->pick = new GeomPick(module);
   geometries[FrameW_SphereDL]->pick->set_highlight(materials[FrameW_HighMatl]);
   geometries[FrameW_SphereDL]->pick->set_cbdata((void*)FrameW_SphereDL);
   geometries[FrameW_SphereDL]->set_matl(materials[FrameW_PointMatl]);
   geometries[FrameW_CylU] = new GeomCylinder;
   geometries[FrameW_CylU]->pick = new GeomPick(module);
   geometries[FrameW_CylU]->pick->set_highlight(materials[FrameW_HighMatl]);
   geometries[FrameW_CylU]->pick->set_cbdata((void*)FrameW_CylU);
   geometries[FrameW_CylU]->set_matl(materials[FrameW_EdgeMatl]);
   geometries[FrameW_CylR] = new GeomCylinder;
   geometries[FrameW_CylR]->pick = new GeomPick(module);
   geometries[FrameW_CylR]->pick->set_highlight(materials[FrameW_HighMatl]);
   geometries[FrameW_CylR]->pick->set_cbdata((void*)FrameW_CylR);
   geometries[FrameW_CylR]->set_matl(materials[FrameW_EdgeMatl]);
   geometries[FrameW_CylD] = new GeomCylinder;
   geometries[FrameW_CylD]->pick = new GeomPick(module);
   geometries[FrameW_CylD]->pick->set_highlight(materials[FrameW_HighMatl]);
   geometries[FrameW_CylD]->pick->set_cbdata((void*)FrameW_CylD);
   geometries[FrameW_CylD]->set_matl(materials[FrameW_EdgeMatl]);
   geometries[FrameW_CylL] = new GeomCylinder;
   geometries[FrameW_CylL]->pick = new GeomPick(module);
   geometries[FrameW_CylL]->pick->set_highlight(materials[FrameW_HighMatl]);
   geometries[FrameW_CylL]->pick->set_cbdata((void*)FrameW_CylL);
   geometries[FrameW_CylL]->set_matl(materials[FrameW_EdgeMatl]);

   widget=new ObjGroup;
   widget->add(geometries[FrameW_SphereUL]);
   widget->add(geometries[FrameW_SphereUR]);
   widget->add(geometries[FrameW_SphereDR]);
   widget->add(geometries[FrameW_SphereDL]);
   widget->add(geometries[FrameW_CylU]);
   widget->add(geometries[FrameW_CylR]);
   widget->add(geometries[FrameW_CylD]);
   widget->add(geometries[FrameW_CylL]);
   widget->pick=new GeomPick(module);

   // Init variables.
   for (Index vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Order();
   
   for (vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Resolve();
}


FrameWidget::~FrameWidget()
{
}


void
FrameWidget::execute()
{
   ((GeomSphere*)geometries[FrameW_SphereUL])->move(variables[FrameW_PointUL]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[FrameW_SphereUR])->move(variables[FrameW_PointUR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[FrameW_SphereDR])->move(variables[FrameW_PointDR]->Get(),
						    1*widget_scale);
   ((GeomSphere*)geometries[FrameW_SphereDL])->move(variables[FrameW_PointDL]->Get(),
						    1*widget_scale);
   ((GeomCylinder*)geometries[FrameW_CylU])->move(variables[FrameW_PointUL]->Get(),
						  variables[FrameW_PointUR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_CylR])->move(variables[FrameW_PointUR]->Get(),
						  variables[FrameW_PointDR]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_CylD])->move(variables[FrameW_PointDR]->Get(),
						  variables[FrameW_PointDL]->Get(),
						  0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_CylL])->move(variables[FrameW_PointDL]->Get(),
						  variables[FrameW_PointUL]->Get(),
						  0.5*widget_scale);

   Vector spvec1(variables[FrameW_PointUR]->Get() - variables[FrameW_PointUL]->Get());
   Vector spvec2(variables[FrameW_PointDL]->Get() - variables[FrameW_PointUL]->Get());
   spvec1.normalize();
   spvec2.normalize();
   Vector v = Cross(spvec1, spvec2);
   geometries[FrameW_SphereUL]->pick->set_principal(spvec1, spvec2, v);
   geometries[FrameW_SphereUR]->pick->set_principal(spvec1, spvec2, v);
   geometries[FrameW_SphereDR]->pick->set_principal(spvec1, spvec2, v);
   geometries[FrameW_SphereDL]->pick->set_principal(spvec1, spvec2, v);
   geometries[FrameW_CylU]->pick->set_principal(spvec1, spvec2, v);
   geometries[FrameW_CylR]->pick->set_principal(spvec1, spvec2, v);
   geometries[FrameW_CylD]->pick->set_principal(spvec1, spvec2, v);
   geometries[FrameW_CylL]->pick->set_principal(spvec1, spvec2, v);
}

void
FrameWidget::geom_moved( int axis, double dist, const Vector& delta,
			 void* cbdata )
{
   cerr << "Moved called..." << endl;
   switch((int)cbdata){
   case FrameW_SphereUL:
      variables[FrameW_PointUL]->SetDelta(delta);
      break;
   case FrameW_SphereUR:
      variables[FrameW_PointUR]->SetDelta(delta);
      break;
   case FrameW_SphereDR:
      variables[FrameW_PointDR]->SetDelta(delta);
      break;
   case FrameW_SphereDL:
      variables[FrameW_PointDL]->SetDelta(delta);
      break;
   case FrameW_CylU:
   case FrameW_CylR:
   case FrameW_CylD:
   case FrameW_CylL:
      variables[FrameW_PointUL]->MoveDelta(delta);
      variables[FrameW_PointUR]->MoveDelta(delta);
      variables[FrameW_PointDR]->MoveDelta(delta);
      variables[FrameW_PointDL]->MoveDelta(delta);
      break;
   }
}

