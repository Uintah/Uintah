
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
const Index NumGeoms = 6;
const Index NumMatls = 4;
const Index NumSchemes = 3;
const Index NumPcks = 6;

enum { GuageW_ConstLine, GuageW_ConstDist, GuageW_ConstSDist, GuageW_ConstRatio };
enum { GuageW_GeomPointL, GuageW_GeomPointR, GuageW_GeomShaft, GuageW_GeomSlider,
       GuageW_GeomResizeL, GuageW_GeomResizeR };
enum { GuageW_PickSphL, GuageW_PickSphR, GuageW_PickCyl, GuageW_PickSlider,
       GuageW_PickResizeL, GuageW_PickResizeR };

GuageWidget::GuageWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale),
  oldaxis(1, 0, 0)
{
   Real INIT = 1.0*widget_scale;
   // Scheme3 is for resizing.
   variables[GuageW_PointL] = new PointVariable("PntL", Scheme1, Point(0, 0, 0));
   variables[GuageW_PointR] = new PointVariable("PntR", Scheme1, Point(INIT, 0, 0));
   variables[GuageW_Slider] = new PointVariable("Slider", Scheme2, Point(INIT/2.0, 0, 0));
   variables[GuageW_Dist] = new RealVariable("Dist", Scheme1, INIT);
   variables[GuageW_SDist] = new RealVariable("SDist", Scheme2, INIT/2.0);
   variables[GuageW_Ratio] = new RealVariable("Ratio", Scheme1, 0.5);
   
   constraints[GuageW_ConstLine] = new SegmentConstraint("ConstLine",
							 NumSchemes,
							 variables[GuageW_PointL],
							 variables[GuageW_PointR],
							 variables[GuageW_Slider]);
   constraints[GuageW_ConstLine]->VarChoices(Scheme1, 2, 2, 2);
   constraints[GuageW_ConstLine]->VarChoices(Scheme2, 2, 2, 2);
   constraints[GuageW_ConstLine]->VarChoices(Scheme3, 2, 2, 2);
   constraints[GuageW_ConstLine]->Priorities(P_Default, P_Default, P_Highest);
   constraints[GuageW_ConstDist] = new DistanceConstraint("ConstDist",
							  NumSchemes,
							  variables[GuageW_PointL],
							  variables[GuageW_PointR],
							  variables[GuageW_Dist]);
   constraints[GuageW_ConstDist]->VarChoices(Scheme1, 1, 0, 1);
   constraints[GuageW_ConstDist]->VarChoices(Scheme2, 1, 0, 1);
   constraints[GuageW_ConstDist]->VarChoices(Scheme3, 2, 2, 2);
   constraints[GuageW_ConstDist]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[GuageW_ConstSDist] = new DistanceConstraint("ConstSDist",
							   NumSchemes,
							   variables[GuageW_PointL],
							   variables[GuageW_Slider],
							   variables[GuageW_SDist]);
   constraints[GuageW_ConstSDist]->VarChoices(Scheme1, 1, 1, 1);
   constraints[GuageW_ConstSDist]->VarChoices(Scheme2, 2, 2, 2);
   constraints[GuageW_ConstSDist]->VarChoices(Scheme3, 1, 1, 1);
   constraints[GuageW_ConstSDist]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[GuageW_ConstRatio] = new RatioConstraint("ConstRatio",
							NumSchemes,
							variables[GuageW_SDist],
							variables[GuageW_Dist],
							variables[GuageW_Ratio]);
   constraints[GuageW_ConstRatio]->VarChoices(Scheme1, 0, 0, 0);
   constraints[GuageW_ConstRatio]->VarChoices(Scheme2, 2, 2, 2);
   constraints[GuageW_ConstRatio]->VarChoices(Scheme3, 0, 0, 0);
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
   GeomGroup* resizes = new GeomGroup;
   geometries[GuageW_GeomResizeL] = new GeomCappedCylinder;
   picks[GuageW_PickResizeL] = new GeomPick(geometries[GuageW_GeomResizeL], module);
   picks[GuageW_PickResizeL]->set_highlight(materials[GuageW_HighMatl]);
   picks[GuageW_PickResizeL]->set_cbdata((void*)GuageW_PickResizeL);
   resizes->add(picks[GuageW_PickResizeL]);
   geometries[GuageW_GeomResizeR] = new GeomCappedCylinder;
   picks[GuageW_PickResizeR] = new GeomPick(geometries[GuageW_GeomResizeR], module);
   picks[GuageW_PickResizeR]->set_highlight(materials[GuageW_HighMatl]);
   picks[GuageW_PickResizeR]->set_cbdata((void*)GuageW_PickResizeR);
   resizes->add(picks[GuageW_PickResizeR]);
   GeomMaterial* resizesm = new GeomMaterial(resizes, SpecialWidgetMaterial);
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
   w->add(resizesm);
   w->add(cylm);
   w->add(sliderm);

   SetEpsilon(widget_scale*1e-6);

   FinishWidget(w);
}


GuageWidget::~GuageWidget()
{
}


void
GuageWidget::widget_execute()
{
   ((GeomSphere*)geometries[GuageW_GeomPointL])->move(variables[GuageW_PointL]->GetPoint(),
						      1*widget_scale);
   ((GeomSphere*)geometries[GuageW_GeomPointR])->move(variables[GuageW_PointR]->GetPoint(),
						      1*widget_scale);
   ((GeomCylinder*)geometries[GuageW_GeomShaft])->move(variables[GuageW_PointL]->GetPoint(),
						       variables[GuageW_PointR]->GetPoint(),
						       0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[GuageW_GeomResizeL])->move(variables[GuageW_PointL]->GetPoint(),
							       variables[GuageW_PointL]->GetPoint()
							       - (GetAxis() * 1.5 * widget_scale),
							       0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[GuageW_GeomResizeR])->move(variables[GuageW_PointR]->GetPoint(),
							       variables[GuageW_PointR]->GetPoint()
							       + (GetAxis() * 1.5 * widget_scale),
							       0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[GuageW_GeomSlider])->move(variables[GuageW_Slider]->GetPoint()
							      - (GetAxis() * 0.3 * widget_scale),
							      variables[GuageW_Slider]->GetPoint()
							      + (GetAxis() * 0.3 * widget_scale),
							      1.1*widget_scale);

   SetEpsilon(widget_scale*1e-6);

   Vector v(GetAxis()), v1, v2;
   v.find_orthogonal(v1,v2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      if ((geom == GuageW_PickSlider) || (geom == GuageW_PickResizeL) || (geom == GuageW_PickResizeR))
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
   case GuageW_PickResizeL:
      variables[GuageW_PointL]->SetDelta(delta, Scheme3);
      break;
   case GuageW_PickResizeR:
      variables[GuageW_PointR]->SetDelta(delta, Scheme3);
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


void
GuageWidget::SetRatio( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[GuageW_Ratio]->Set(ratio);
   
   execute();
}


Real
GuageWidget::GetRatio() const
{
   return (variables[GuageW_Ratio]->GetReal());
}


void
GuageWidget::SetEndpoints( const Point& end1, const Point& end2 )
{
   variables[GuageW_PointL]->Move(end1);
   variables[GuageW_PointR]->Set(end2);
   
   execute();
}


void
GuageWidget::GetEndpoints( Point& end1, Point& end2 ) const
{
   end1 = variables[GuageW_PointL]->GetPoint();
   end2 = variables[GuageW_PointR]->GetPoint();
}


const Vector&
GuageWidget::GetAxis()
{
   Vector axis(variables[GuageW_PointR]->GetPoint() - variables[GuageW_PointL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis;
   else 
      return (oldaxis = axis.normal());
}


