
/*
 *  GuageWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Widgets/GuageWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/SegmentConstraint.h>


const Index NumCons = 2;
const Index NumVars = 4;
const Index NumGeoms = 4;
const Index NumMatls = 3;
const Index NumSchemes = 2;

enum { GuageW_ConstLine, GuageW_ConstDist };
enum { GuageW_SphereL, GuageW_SphereR, GuageW_Cylinder, GuageW_SliderCyl };
enum { GuageW_PointMatl, GuageW_EdgeMatl, GuageW_HighMatl };

GuageWidget::GuageWidget( Module* module )
: BaseWidget(module, NumVars, NumCons, NumGeoms, NumMatls)
{
   const Real INIT = 100.0;
   variables[GuageW_PointL] = new Variable("PntL", Scheme1, Point(0, 0, 0));
   variables[GuageW_PointR] = new Variable("PntR", Scheme1, Point(INIT, 0, 0));
   variables[GuageW_Slider] = new Variable("Slider", Scheme2, Point(INIT/2.0, 0, 0));
   variables[GuageW_Dist] = new Variable("Dist", Scheme1, Point(INIT/2.0, 0, 0));

   NOT_FINISHED("Constraints not right!");
   
   constraints[GuageW_ConstLine] = new SegmentConstraint("ConstLine",
							 NumSchemes,
							 variables[GuageW_PointL],
							 variables[GuageW_PointR],
							 variables[GuageW_Slider]);
   constraints[GuageW_ConstLine]->VarChoices(Scheme1, 2, 2, 1);
   constraints[GuageW_ConstLine]->VarChoices(Scheme2, 1, 0, 1);
   constraints[GuageW_ConstLine]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[GuageW_ConstDist] = new DistanceConstraint("ConstDist",
							  NumSchemes,
							  variables[GuageW_PointL],
							  variables[GuageW_Slider],
							  variables[GuageW_Dist]);
   constraints[GuageW_ConstDist]->VarChoices(Scheme1, 1, 0, 1);
   constraints[GuageW_ConstDist]->VarChoices(Scheme2, 2, 2, 1);
   constraints[GuageW_ConstDist]->Priorities(P_Highest, P_Highest, P_Default);

   materials[GuageW_PointMatl] = new MaterialProp(Color(0,0,0), Color(.54, .60, 1),
						  Color(.5,.5,.5), 20);
   materials[GuageW_EdgeMatl] = new MaterialProp(Color(0,0,0), Color(.54, .60, .66),
						 Color(.5,.5,.5), 20);
   materials[GuageW_HighMatl] = new MaterialProp(Color(0,0,0), Color(.7,.7,.7),
						 Color(0,0,.6), 20);

   Index geom;
   for (geom = GuageW_SphereL; geom <= GuageW_SphereL; geom++) {
      geometries[geom] = new GeomSphere;
      geometries[geom]->pick = new GeomPick(module);
      geometries[geom]->pick->set_highlight(materials[GuageW_HighMatl]);
      geometries[geom]->pick->set_cbdata((void*)geom);
      geometries[geom]->set_matl(materials[GuageW_PointMatl]);
   }
   for (geom = GuageW_Cylinder; geom <= GuageW_SliderCyl; geom++) {
      geometries[geom] = new GeomCylinder;
      geometries[geom]->pick = new GeomPick(module);
      geometries[geom]->pick->set_highlight(materials[GuageW_HighMatl]);
      geometries[geom]->pick->set_cbdata((void*)geom);
      geometries[geom]->set_matl(materials[GuageW_EdgeMatl]);
   }

   widget = new ObjGroup;
   for (geom = 0; geom <= NumGeoms; geom++) {
      widget->add(geometries[geom]);
   }
   widget->pick=new GeomPick(module);

   // Init variables.
   for (Index vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Order();
   
   for (vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Resolve();
}


GuageWidget::~GuageWidget()
{
}


void
GuageWidget::execute()
{
   ((GeomSphere*)geometries[GuageW_SphereL])->move(variables[GuageW_PointL]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[GuageW_SphereR])->move(variables[GuageW_PointR]->Get(),
						   1*widget_scale);
   ((GeomCylinder*)geometries[GuageW_Cylinder])->move(variables[GuageW_PointL]->Get(),
						      variables[GuageW_PointR]->Get(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[GuageW_SliderCyl])->move(variables[GuageW_PointL]->Get(),
						       variables[GuageW_Slider]->Get(),
						       0.75*widget_scale);

   Vector v(variables[GuageW_PointR]->Get() - variables[GuageW_PointL]->Get());
   v.normalize();
   for (Index geom = 0; geom <= NumGeoms; geom++) {
      geometries[geom]->pick->set_principal(v);
   }
}

void
GuageWidget::geom_moved( int axis, double dist, const Vector& delta,
			 void* cbdata )
{
   cerr << "Moved called..." << endl;
   switch((int)cbdata){
   case GuageW_SphereL:
      variables[GuageW_PointL]->SetDelta(delta);
      break;
   case GuageW_SphereR:
      variables[GuageW_PointR]->SetDelta(delta);
      break;
   case GuageW_SliderCyl:
      variables[GuageW_Slider]->SetDelta(delta);
      break;
   case GuageW_Cylinder:
      variables[GuageW_PointL]->MoveDelta(delta);
      variables[GuageW_PointR]->MoveDelta(delta);
      variables[GuageW_Slider]->MoveDelta(delta);
      break;
   }
}

