
/*
 *  GaugeWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Widgets/GaugeWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/SegmentConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 4;
const Index NumVars = 6;
const Index NumGeoms = 6;
const Index NumPcks = 6;
const Index NumMdes = 2;
const Index NumSwtchs = 2;
const Index NumSchemes = 3;

enum { ConstLine, ConstDist, ConstSDist, ConstRatio };
enum { GeomPointL, GeomPointR, GeomShaft, GeomSlider,
       GeomResizeL, GeomResizeR };
enum { PickSphL, PickSphR, PickCyl, PickSlider,
       PickResizeL, PickResizeR };

GaugeWidget::GaugeWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumPcks, NumMdes, NumSwtchs, widget_scale),
  oldaxis(1, 0, 0)
{
   Real INIT = 1.0*widget_scale;
   // Scheme3 is for resizing.
   variables[PointLVar] = new PointVariable("PntL", solve, Scheme1, Point(0, 0, 0));
   variables[PointRVar] = new PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
   variables[SliderVar] = new PointVariable("Slider", solve, Scheme2, Point(INIT/2.0, 0, 0));
   variables[DistVar] = new RealVariable("Dist", solve, Scheme1, INIT);
   variables[SDistVar] = new RealVariable("SDistVar", solve, Scheme2, INIT/2.0);
   variables[RatioVar] = new RealVariable("Ratio", solve, Scheme1, 0.5);
   
   constraints[ConstLine] = new SegmentConstraint("ConstLine",
						  NumSchemes,
						  variables[PointLVar],
						  variables[PointRVar],
						  variables[SliderVar]);
   constraints[ConstLine]->VarChoices(Scheme1, 2, 2, 2);
   constraints[ConstLine]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstLine]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstLine]->Priorities(P_Default, P_Default, P_Highest);
   constraints[ConstDist] = new DistanceConstraint("ConstDist",
						   NumSchemes,
						   variables[PointLVar],
						   variables[PointRVar],
						   variables[DistVar]);
   constraints[ConstDist]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstDist]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstDist]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstDist]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstSDist] = new DistanceConstraint("ConstSDist",
						    NumSchemes,
						    variables[PointLVar],
						    variables[SliderVar],
						    variables[SDistVar]);
   constraints[ConstSDist]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstSDist]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstSDist]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstSDist]->Priorities(P_Lowest, P_Default, P_Default);
   constraints[ConstRatio] = new RatioConstraint("ConstRatio",
						 NumSchemes,
						 variables[SDistVar],
						 variables[DistVar],
						 variables[RatioVar]);
   constraints[ConstRatio]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatio]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstRatio]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRatio]->Priorities(P_Highest, P_Highest, P_Highest);

   geometries[GeomShaft] = new GeomCappedCylinder;
   picks[PickCyl] = new GeomPick(geometries[GeomShaft], module, this, PickCyl);
   picks[PickCyl]->set_highlight(HighlightMaterial);
   GeomMaterial* cylm = new GeomMaterial(picks[PickCyl], EdgeMaterial);   
   CreateModeSwitch(0, cylm);

   geometries[GeomPointL] = new GeomSphere;
   picks[PickSphL] = new GeomPick(geometries[GeomPointL],
				  module, this, PickSphL);
   picks[PickSphL]->set_highlight(HighlightMaterial);
   GeomMaterial* sphlm = new GeomMaterial(picks[PickSphL], PointMaterial);
   geometries[GeomPointR] = new GeomSphere;
   picks[PickSphR] = new GeomPick(geometries[GeomPointR],
				  module, this, PickSphR);
   picks[PickSphR]->set_highlight(HighlightMaterial);
   GeomMaterial* sphrm = new GeomMaterial(picks[PickSphR], PointMaterial);
   GeomGroup* resizes = new GeomGroup;
   geometries[GeomResizeL] = new GeomCappedCylinder;
   picks[PickResizeL] = new GeomPick(geometries[GeomResizeL],
					    module, this, PickResizeL);
   picks[PickResizeL]->set_highlight(HighlightMaterial);
   resizes->add(picks[PickResizeL]);
   geometries[GeomResizeR] = new GeomCappedCylinder;
   picks[PickResizeR] = new GeomPick(geometries[GeomResizeR],
				     module, this, PickResizeR);
   picks[PickResizeR]->set_highlight(HighlightMaterial);
   resizes->add(picks[PickResizeR]);
   GeomMaterial* resizesm = new GeomMaterial(resizes, ResizeMaterial);
   
   geometries[GeomSlider] = new GeomCappedCylinder;
   picks[PickSlider] = new GeomPick(geometries[GeomSlider],
				    module, this, PickSlider);
   picks[PickSlider]->set_highlight(HighlightMaterial);
   GeomMaterial* sliderm = new GeomMaterial(picks[PickSlider], SliderMaterial);
   GeomGroup* w = new GeomGroup;
   w->add(sphlm);
   w->add(sphrm);
   w->add(resizesm);
   w->add(sliderm);
   CreateModeSwitch(1, w);

   SetMode(Mode0, Switch0|Switch1);
   SetMode(Mode1, Switch0);

   FinishWidget();
}


GaugeWidget::~GaugeWidget()
{
}


void
GaugeWidget::widget_execute()
{
   Point L(variables[PointLVar]->point()), R(variables[PointRVar]->point()), S(variables[SliderVar]->point());

   if (mode_switches[0]->get_state()) {
      ((GeomCappedCylinder*)geometries[GeomShaft])->move(L, R, 0.5*widget_scale);
   }

   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[GeomPointL])->move(L, widget_scale);
      ((GeomSphere*)geometries[GeomPointR])->move(R, widget_scale);
      ((GeomCappedCylinder*)geometries[GeomResizeL])->move(L, L - (GetAxis() * 1.5 * widget_scale),
							   0.5*widget_scale);
      ((GeomCappedCylinder*)geometries[GeomResizeR])->move(R, R + (GetAxis() * 1.5 * widget_scale),
							   0.5*widget_scale);
      ((GeomCappedCylinder*)geometries[GeomSlider])->move(S - (GetAxis() * 0.3 * widget_scale),
							  S + (GetAxis() * 0.3 * widget_scale),
							  1.1*widget_scale);
   }
   
   Vector v(GetAxis()), v1, v2;
   v.find_orthogonal(v1,v2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      if ((geom == PickSlider) || (geom == PickResizeL) || (geom == PickResizeR))
	 picks[geom]->set_principal(v);
      else
	 picks[geom]->set_principal(v, v1, v2);
   }
}


void
GaugeWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 int cbdata )
{
   ((DistanceConstraint*)constraints[ConstSDist])->SetDefault(GetAxis());
   ((DistanceConstraint*)constraints[ConstDist])->SetDefault(GetAxis());

   switch(cbdata){
   case PickSphL:
      variables[PointLVar]->SetDelta(delta);
      break;
   case PickSphR:
      variables[PointRVar]->SetDelta(delta);
      break;
   case PickResizeL:
      variables[PointLVar]->SetDelta(delta, Scheme3);
      break;
   case PickResizeR:
      variables[PointRVar]->SetDelta(delta, Scheme3);
      break;
   case PickSlider:
      variables[SliderVar]->SetDelta(delta);
      break;
   case PickCyl:
      MoveDelta(delta);
      break;
   }
   execute();
}


void
GaugeWidget::MoveDelta( const Vector& delta )
{
   variables[PointLVar]->MoveDelta(delta);
   variables[PointRVar]->MoveDelta(delta);
   variables[SliderVar]->MoveDelta(delta);

   execute();
}


Point
GaugeWidget::ReferencePoint() const
{
   return (variables[PointLVar]->point()
	   + (variables[PointRVar]->point()
	      -variables[PointLVar]->point())/2.0);
}


void
GaugeWidget::SetRatio( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[RatioVar]->Set(ratio);
   
   execute();
}


Real
GaugeWidget::GetRatio() const
{
   return (variables[RatioVar]->real());
}


void
GaugeWidget::SetEndpoints( const Point& end1, const Point& end2 )
{
   variables[PointLVar]->Move(end1);
   variables[PointRVar]->Set(end2);
   
   execute();
}


void
GaugeWidget::GetEndpoints( Point& end1, Point& end2 ) const
{
   end1 = variables[PointLVar]->point();
   end2 = variables[PointRVar]->point();
}


const Vector&
GaugeWidget::GetAxis()
{
   Vector axis(variables[PointRVar]->point() - variables[PointLVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis;
   else 
      return (oldaxis = axis.normal());
}


