
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
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 4;
const Index NumVars = 6;
const Index NumGeoms = 4;
const Index NumMatls = 4;
const Index NumSchemes = 2;
const Index NumPcks = 4;

enum { GuageW_ConstLine, GuageW_ConstDist, GuageW_ConstSDist, GuageW_ConstRatio };
enum { GuageW_GeomPointL, GuageW_GeomPointR, GuageW_GeomShaft, GuageW_GeomSlider };
enum { GuageW_PickSphL, GuageW_PickSphR, GuageW_PickCyl, GuageW_PickSlider };

GuageWidget::GuageWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale),
  oldaxis(1, 0, 0)
{
   Real INIT = 1.0*widget_scale;
   variables[GuageW_PointL] = new Variable("PntL", Scheme1, Point(0, 0, 0));
   variables[GuageW_PointR] = new Variable("PntR", Scheme1, Point(INIT, 0, 0));
   variables[GuageW_Slider] = new Variable("Slider", Scheme2, Point(INIT/2.0, 0, 0));
   variables[GuageW_Dist] = new Variable("Dist", Scheme1, Point(INIT, 0, 0));
   variables[GuageW_SDist] = new Variable("SDist", Scheme2, Point(INIT/2.0, 0, 0));
   variables[GuageW_Ratio] = new Variable("Ratio", Scheme1, Point(0.5, 0, 0));
   
   constraints[GuageW_ConstLine] = new SegmentConstraint("ConstLine",
							 NumSchemes,
							 variables[GuageW_PointL],
							 variables[GuageW_PointR],
							 variables[GuageW_Slider]);
   constraints[GuageW_ConstLine]->VarChoices(Scheme1, 2, 2, 2);
   constraints[GuageW_ConstLine]->VarChoices(Scheme2, 2, 2, 2);
   constraints[GuageW_ConstLine]->Priorities(P_Default, P_Default, P_Highest);
   constraints[GuageW_ConstDist] = new DistanceConstraint("ConstDist",
							  NumSchemes,
							  variables[GuageW_PointL],
							  variables[GuageW_PointR],
							  variables[GuageW_Dist]);
   constraints[GuageW_ConstDist]->VarChoices(Scheme1, 2, 2, 2);
   constraints[GuageW_ConstDist]->VarChoices(Scheme2, 2, 2, 2);
   constraints[GuageW_ConstDist]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[GuageW_ConstSDist] = new DistanceConstraint("ConstSDist",
							  NumSchemes,
							  variables[GuageW_PointL],
							  variables[GuageW_Slider],
							  variables[GuageW_SDist]);
   constraints[GuageW_ConstSDist]->VarChoices(Scheme1, 1, 1, 1);
   constraints[GuageW_ConstSDist]->VarChoices(Scheme2, 2, 2, 2);
   constraints[GuageW_ConstSDist]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[GuageW_ConstRatio] = new RatioConstraint("ConstRatio",
							NumSchemes,
							variables[GuageW_SDist],
							variables[GuageW_Dist],
							variables[GuageW_Ratio]);
   constraints[GuageW_ConstRatio]->VarChoices(Scheme1, 0, 0, 0);
   constraints[GuageW_ConstRatio]->VarChoices(Scheme2, 2, 2, 2);
   constraints[GuageW_ConstRatio]->Priorities(P_Highest, P_Highest, P_Highest);

   materials[GuageW_PointMatl] = PointWidgetMaterial;
   materials[GuageW_EdgeMatl] = EdgeWidgetMaterial;
   materials[GuageW_SliderMatl] = SliderWidgetMaterial;
   materials[GuageW_HighMatl] = HighlightWidgetMaterial;

   geometries[GuageW_GeomPointL] = new GeomSphere;
   picks[GuageW_PickSphL] = new GeomPick(geometries[GuageW_GeomPointL], module);
   picks[GuageW_PickSphL]->set_highlight(materials[GuageW_HighMatl]);
   picks[GuageW_PickSphL]->set_cbdata((void*)GuageW_PickSphL);
   GeomMaterial* sphlm = new GeomMaterial(picks[GuageW_PickSphL], materials[GuageW_PointMatl]);
   geometries[GuageW_GeomPointR] = new GeomSphere;
   picks[GuageW_PickSphR] = new GeomPick(geometries[GuageW_GeomPointR], module);
   picks[GuageW_PickSphR]->set_highlight(materials[GuageW_HighMatl]);
   picks[GuageW_PickSphR]->set_cbdata((void*)GuageW_PickSphR);
   GeomMaterial* sphrm = new GeomMaterial(picks[GuageW_PickSphR], materials[GuageW_PointMatl]);
   geometries[GuageW_GeomShaft] = new GeomCylinder;
   picks[GuageW_PickCyl] = new GeomPick(geometries[GuageW_GeomShaft], module);
   picks[GuageW_PickCyl]->set_highlight(materials[GuageW_HighMatl]);
   picks[GuageW_PickCyl]->set_cbdata((void*)GuageW_PickCyl);
   GeomMaterial* cylm = new GeomMaterial(picks[GuageW_PickCyl], materials[GuageW_EdgeMatl]);
   geometries[GuageW_GeomSlider] = new GeomCappedCylinder;
   picks[GuageW_PickSlider] = new GeomPick(geometries[GuageW_GeomSlider], module);
   picks[GuageW_PickSlider]->set_highlight(materials[GuageW_HighMatl]);
   picks[GuageW_PickSlider]->set_cbdata((void*)GuageW_PickSlider);
   GeomMaterial* sliderm = new GeomMaterial(picks[GuageW_PickSlider], materials[GuageW_SliderMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(sphlm);
   w->add(sphrm);
   w->add(cylm);
   w->add(sliderm);

   SetEpsilon(widget_scale*1e-4);

   FinishWidget(w);
}


GuageWidget::~GuageWidget()
{
}


void
GuageWidget::widget_execute()
{
   ((GeomSphere*)geometries[GuageW_GeomPointL])->move(variables[GuageW_PointL]->Get(),
						   1*widget_scale);
   ((GeomSphere*)geometries[GuageW_GeomPointR])->move(variables[GuageW_PointR]->Get(),
						   1*widget_scale);
   ((GeomCylinder*)geometries[GuageW_GeomShaft])->move(variables[GuageW_PointL]->Get(),
						      variables[GuageW_PointR]->Get(),
						      0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[GuageW_GeomSlider])->move(variables[GuageW_Slider]->Get()
							     - (GetAxis() * 0.3 * widget_scale),
							     variables[GuageW_Slider]->Get()
							     + (GetAxis() * 0.3 * widget_scale),
							     1.1*widget_scale);

   Vector v(GetAxis()), v1, v2;
   v.find_orthogonal(v1,v2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      if (geom == GuageW_PickSlider)
	 picks[geom]->set_principal(v);
      else
	 picks[geom]->set_principal(v, v1, v2);
   }
}

void
GuageWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   ((DistanceConstraint*)constraints[GuageW_ConstSDist])->SetDefault(GetAxis());
   ((DistanceConstraint*)constraints[GuageW_ConstDist])->SetDefault(GetAxis());

   for (Index v=0; v<NumVars; v++)
      variables[v]->Reset();
   
   switch((int)cbdata){
   case GuageW_PickSphL:
      variables[GuageW_PointL]->SetDelta(delta);
      break;
   case GuageW_PickSphR:
      variables[GuageW_PointR]->SetDelta(delta);
      break;
   case GuageW_PickSlider:
      variables[GuageW_Slider]->SetDelta(delta);
      break;
   case GuageW_PickCyl:
      variables[GuageW_PointL]->MoveDelta(delta);
      variables[GuageW_PointR]->MoveDelta(delta);
      variables[GuageW_Slider]->MoveDelta(delta);
      break;
   }
}

