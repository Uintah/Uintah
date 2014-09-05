//static char *id="@(#) $Id$";

/*
 *  ScaledFrameWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <PSECore/Widgets/ScaledFrameWidget.h>
#include <PSECore/Constraints/DistanceConstraint.h>
#include <PSECore/Constraints/PythagorasConstraint.h>
#include <PSECore/Constraints/RatioConstraint.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Dataflow/Module.h>

namespace PSECore {
namespace Widgets {

using SCICore::GeomSpace::GeomGroup;
using SCICore::GeomSpace::GeomSphere;
using SCICore::GeomSpace::GeomCylinder;
using SCICore::GeomSpace::GeomCappedCylinder;
using SCICore::Geometry::Cross;

using namespace PSECore::Constraints;

const Index NumCons = 6;
const Index NumVars = 10;
const Index NumGeoms = 18;
const Index NumPcks = 11;
const Index NumMatls = 4;
const Index NumMdes = 7;
const Index NumSwtchs = 4;
const Index NumSchemes = 5;

enum { ConstRD, ConstRC, ConstDC, ConstPyth,
       ConstRatioR, ConstRatioD };
enum { GeomSPointUL, GeomSPointUR, GeomSPointDR, GeomSPointDL,
       GeomPointU, GeomPointR, GeomPointD, GeomPointL,
       GeomCylU, GeomCylR, GeomCylD, GeomCylL,
       GeomResizeU, GeomResizeR, GeomResizeD, GeomResizeL,
       GeomSliderCylR, GeomSliderCylD };
enum { PickSphU, PickSphR, PickSphD, PickSphL, PickCyls,
       PickResizeU, PickResizeR, PickResizeD, PickResizeL,
       PickSliderR, PickSliderD };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
ScaledFrameWidget::ScaledFrameWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, "ScaledFrameWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  oldrightaxis(1, 0, 0), olddownaxis(0, 1, 0)
{
   Real INIT = 5.0*widget_scale;
   variables[CenterVar] = scinew PointVariable("Center", solve, Scheme1, Point(0, 0, 0));
   variables[PointRVar] = scinew PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
   variables[PointDVar] = scinew PointVariable("PntD", solve, Scheme2, Point(0, INIT, 0));
   variables[DistRVar] = scinew RealVariable("RDIST", solve, Scheme3, INIT);
   variables[DistDVar] = scinew RealVariable("DDIST", solve, Scheme4, INIT);
   variables[HypoVar] = scinew RealVariable("HYPO", solve, Scheme3, sqrt(2*INIT*INIT));
   variables[SDistRVar] = scinew RealVariable("SDistR", solve, Scheme5, INIT/2.0);
   variables[SDistDVar] = scinew RealVariable("SDistD", solve, Scheme5, INIT/2.0);
   variables[RatioRVar] = scinew RealVariable("RatioR", solve, Scheme1, 0.5);
   variables[RatioDVar] = scinew RealVariable("RatioD", solve, Scheme1, 0.5);

   constraints[ConstRatioR] = scinew RatioConstraint("ConstRatioR",
						  NumSchemes,
						  variables[SDistRVar],
						  variables[DistRVar],
						  variables[RatioRVar]);
   constraints[ConstRatioR]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstRatioR]->VarChoices(Scheme5, 2, 2, 2);
   constraints[ConstRatioR]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRatioD] = scinew RatioConstraint("ConstRatioD",
						  NumSchemes,
						  variables[SDistDVar],
						  variables[DistDVar],
						  variables[RatioDVar]);
   constraints[ConstRatioD]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstRatioD]->VarChoices(Scheme5, 2, 2, 2);
   constraints[ConstRatioD]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRD] = scinew DistanceConstraint("ConstRD",
						 NumSchemes,
						 variables[PointRVar],
						 variables[PointDVar],
						 variables[HypoVar]);
   constraints[ConstRD]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstRD]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRD]->VarChoices(Scheme3, 2, 2, 1);
   constraints[ConstRD]->VarChoices(Scheme4, 2, 2, 0);
   constraints[ConstRD]->VarChoices(Scheme5, 1, 0, 1);
   constraints[ConstRD]->Priorities(P_Default, P_Default, P_Default);
   constraints[ConstPyth] = scinew PythagorasConstraint("ConstPyth",
						     NumSchemes,
						     variables[DistRVar],
						     variables[DistDVar],
						     variables[HypoVar]);
   constraints[ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPyth]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstPyth]->VarChoices(Scheme3, 2, 2, 1);
   constraints[ConstPyth]->VarChoices(Scheme4, 2, 2, 0);
   constraints[ConstPyth]->VarChoices(Scheme5, 1, 0, 1);
   constraints[ConstPyth]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRC] = scinew DistanceConstraint("ConstRC",
						 NumSchemes,
						 variables[PointRVar],
						 variables[CenterVar],
						 variables[DistRVar]);
   constraints[ConstRC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstRC]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme5, 1, 0, 1);
   constraints[ConstRC]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstDC] = scinew DistanceConstraint("ConstDC",
					       NumSchemes,
					       variables[PointDVar],
					       variables[CenterVar],
					       variables[DistDVar]);
   constraints[ConstDC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstDC]->VarChoices(Scheme5, 1, 0, 1);
   constraints[ConstDC]->Priorities(P_Highest, P_Highest, P_Default);

   Index geom, pick;
   GeomGroup* cyls = scinew GeomGroup;
   for (geom = GeomSPointUL; geom <= GeomSPointDL; geom++) {
      geometries[geom] = scinew GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomCylU; geom <= GeomCylL; geom++) {
      geometries[geom] = scinew GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[PickCyls] = scinew GeomPick(cyls, module, this, PickCyls);
   picks[PickCyls]->set_highlight(DefaultHighlightMaterial);
   materials[EdgeMatl] = scinew GeomMaterial(picks[PickCyls], DefaultEdgeMaterial);
   CreateModeSwitch(0, materials[EdgeMatl]);

   GeomGroup* pts = scinew GeomGroup;
   for (geom = GeomPointU, pick = PickSphU;
	geom <= GeomPointL; geom++, pick++) {
      geometries[geom] = scinew GeomSphere;
      picks[pick] = scinew GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(DefaultHighlightMaterial);
      pts->add(picks[pick]);
   }
   materials[PointMatl] = scinew GeomMaterial(pts, DefaultPointMaterial);
   CreateModeSwitch(1, materials[PointMatl]);
   
   GeomGroup* resizes = scinew GeomGroup;
   for (geom = GeomResizeU, pick = PickResizeU;
	geom <= GeomResizeL; geom++, pick++) {
      geometries[geom] = scinew GeomCappedCylinder;
      picks[pick] = scinew GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(DefaultHighlightMaterial);
      resizes->add(picks[pick]);
   }
   materials[ResizeMatl] = scinew GeomMaterial(resizes, DefaultResizeMaterial);
   CreateModeSwitch(2, materials[ResizeMatl]);

   GeomGroup* sliders = scinew GeomGroup;
   geometries[GeomSliderCylR] = scinew GeomCappedCylinder;
   picks[PickSliderR] = scinew GeomPick(geometries[GeomSliderCylR], module, this, PickSliderR);
   picks[PickSliderR]->set_highlight(DefaultHighlightMaterial);
   sliders->add(picks[PickSliderR]);
   geometries[GeomSliderCylD] = scinew GeomCappedCylinder;
   picks[PickSliderD] = scinew GeomPick(geometries[GeomSliderCylD], module, this, PickSliderD);
   picks[PickSliderD]->set_highlight(DefaultHighlightMaterial);
   sliders->add(picks[PickSliderD]);
   materials[SliderMatl] = scinew GeomMaterial(sliders, DefaultSliderMaterial);
   CreateModeSwitch(3, materials[SliderMatl]);

   // Switch0 are the bars
   // Switch1 are the rotation points
   // Switch2 are the resize cylinders
   // Switch3 are the sliders
   SetMode(Mode0, Switch0|Switch1|Switch2|Switch3);
   SetMode(Mode1, Switch0|Switch1|Switch2);
   SetMode(Mode2, Switch0|Switch3);
   SetMode(Mode3, Switch0);
   SetMode(Mode4, Switch0|Switch1|Switch3);
   SetMode(Mode5, Switch0|Switch2|Switch3);
   SetMode(Mode6, Switch0|Switch2);
   
   FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
ScaledFrameWidget::~ScaledFrameWidget()
{
}


/***************************************************************************
 * The widget's redraw method changes widget geometry to reflect the
 *      widget's variable values and its widget_scale.
 * Geometry should only be changed if the mode_switch that displays
 *      that geometry is active.
 * Redraw should also set the principal directions for all picks.
 * Redraw should never be called directly; the BaseWidget execute method
 *      calls redraw after establishing the appropriate locks.
 */
void
ScaledFrameWidget::redraw()
{
   Real sphererad(widget_scale), resizerad(0.5*widget_scale), cylinderrad(0.5*widget_scale);
   Vector Right(GetRightAxis()*variables[DistRVar]->real());
   Vector Down(GetDownAxis()*variables[DistDVar]->real());
   Point Center(variables[CenterVar]->point());
   Point UL(Center-Right-Down);
   Point UR(Center+Right-Down);
   Point DR(Center+Right+Down);
   Point DL(Center-Right+Down);
   Point U(Center-Down);
   Point R(Center+Right);
   Point D(Center+Down);
   Point L(Center-Right);

   // draw the bars
   if (mode_switches[0]->get_state()) {
      ((GeomCylinder*)geometries[GeomCylU])->move(UL, UR, cylinderrad);
      ((GeomCylinder*)geometries[GeomCylR])->move(UR, DR, cylinderrad);
      ((GeomCylinder*)geometries[GeomCylD])->move(DR, DL, cylinderrad);
      ((GeomCylinder*)geometries[GeomCylL])->move(DL, UL, cylinderrad);
      ((GeomSphere*)geometries[GeomSPointUL])->move(UL, cylinderrad);
      ((GeomSphere*)geometries[GeomSPointUR])->move(UR, cylinderrad);
      ((GeomSphere*)geometries[GeomSPointDR])->move(DR, cylinderrad);
      ((GeomSphere*)geometries[GeomSPointDL])->move(DL, cylinderrad);
   }

   // draw the rotation points
   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[GeomPointU])->move(U, sphererad);
      ((GeomSphere*)geometries[GeomPointR])->move(R, sphererad);
      ((GeomSphere*)geometries[GeomPointD])->move(D, sphererad);
      ((GeomSphere*)geometries[GeomPointL])->move(L, sphererad);
   }

   // draw the resize cylinders
   if (mode_switches[2]->get_state()) {
      Vector resizeR(GetRightAxis()*1.5*widget_scale),
	 resizeD(GetDownAxis()*1.5*widget_scale);
      
      ((GeomCappedCylinder*)geometries[GeomResizeU])->move(U, U - resizeD, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeR])->move(R, R + resizeR, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeD])->move(D, D + resizeD, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeL])->move(L, L - resizeR, resizerad);
   }

   // draw the sliders
   if (mode_switches[3]->get_state()) {
      Point SliderR(UL+GetRightAxis()*variables[SDistRVar]->real()*2.0);
      Point SliderD(UL+GetDownAxis()*variables[SDistDVar]->real()*2.0);
      ((GeomCappedCylinder*)geometries[GeomSliderCylR])->move(SliderR - (GetRightAxis() * 0.3 * widget_scale),
							      SliderR + (GetRightAxis() * 0.3 * widget_scale),
							      1.1*widget_scale);
      ((GeomCappedCylinder*)geometries[GeomSliderCylD])->move(SliderD - (GetDownAxis() * 0.3 * widget_scale),
							      SliderD + (GetDownAxis() * 0.3 * widget_scale),
							      1.1*widget_scale);
   }

   ((DistanceConstraint*)constraints[ConstRC])->SetMinimum(1.6*widget_scale);
   ((DistanceConstraint*)constraints[ConstDC])->SetMinimum(1.6*widget_scale);
   ((DistanceConstraint*)constraints[ConstRD])->SetMinimum(sqrt(2*1.6*1.6)*widget_scale);

   Right.normalize();
   Down.normalize();
   Vector Norm(Cross(Right, Down));
   for (Index geom = 0; geom < NumPcks; geom++) {
      if ((geom == PickResizeU) || (geom == PickResizeD) || (geom == PickSliderD))
	 picks[geom]->set_principal(Down);
      else if ((geom == PickResizeL) || (geom == PickResizeR) || (geom == PickSliderR))
	 picks[geom]->set_principal(Right);
      else if ((geom == PickSphU) || (geom == PickSphD))
	 picks[geom]->set_principal(Right, Norm);
      else if ((geom == PickSphL) || (geom == PickSphR))
	 picks[geom]->set_principal(Down, Norm);
      else
	 picks[geom]->set_principal(Right, Down, Norm);
   }
}

/***************************************************************************
 * The widget's geom_moved method receives geometry move requests from
 *      the widget's picks.  The widget's variables must be altered to
 *      reflect these changes based upon which pick made the request.
 * No more than one variable should be Set since this triggers solution of
 *      the constraints--multiple Sets could lead to inconsistencies.
 *      The constraint system only requires that a variable be Set if the
 *      change would cause a constraint to be invalid.  For example, if
 *      all PointVariables are moved by the same delta, then no Set is
 *      required.
 * The last line of the widget's geom_moved method should call the
 *      BaseWidget execute method (which calls the redraw method).
 */
void
ScaledFrameWidget::geom_moved( GeomPick*, int axis, double dist,
			       const Vector& delta, int pick, const BState& )
{
   Vector delt(delta);
   Real ResizeMin(1.5*widget_scale);
   if (axis==1) dist = -dist;

   ((DistanceConstraint*)constraints[ConstRC])->SetDefault(GetRightAxis());
   ((DistanceConstraint*)constraints[ConstDC])->SetDefault(GetDownAxis());

   switch(pick){
   case PickSphU:
      variables[PointDVar]->SetDelta(-delta);
      break;
   case PickSphR:
      variables[PointRVar]->SetDelta(delta);
      break;
   case PickSphD:
      variables[PointDVar]->SetDelta(delta);
      break;
   case PickSphL:
      variables[PointRVar]->SetDelta(-delta);
      break;
   case PickResizeU:
      if ((variables[DistDVar]->real() - dist) < ResizeMin)
	 delt = variables[CenterVar]->point() + GetDownAxis()*ResizeMin - variables[PointDVar]->point();
      variables[PointRVar]->MoveDelta(delt/2.0);      
      variables[CenterVar]->SetDelta(delt/2.0, Scheme4);
      break;
   case PickResizeR:
      if ((variables[DistRVar]->real() + dist) < ResizeMin)
	 delt = variables[CenterVar]->point() + GetDownAxis()*ResizeMin - variables[PointDVar]->point();
      variables[CenterVar]->MoveDelta(delt/2.0);
      variables[PointDVar]->MoveDelta(delt/2.0);      
      variables[PointRVar]->SetDelta(delt, Scheme3);
      break;
   case PickResizeD:
      if ((variables[DistDVar]->real() + dist) < ResizeMin)
	 delt = variables[CenterVar]->point() + GetDownAxis()*ResizeMin - variables[PointDVar]->point();
      variables[CenterVar]->MoveDelta(delt/2.0);
      variables[PointRVar]->MoveDelta(delt/2.0);      
      variables[PointDVar]->SetDelta(delt, Scheme4);
      break;
   case PickResizeL:
      if ((variables[DistRVar]->real() - dist) < ResizeMin)
	 delt = variables[CenterVar]->point() + GetDownAxis()*ResizeMin - variables[PointDVar]->point();
      variables[PointDVar]->MoveDelta(delt/2.0);      
      variables[CenterVar]->SetDelta(delt/2.0, Scheme3);
      break;
   case PickSliderR:
      {
	  Real sdist(variables[SDistRVar]->real()+dist/2.0);
	  if (sdist<0.0) sdist=0.0;
	  else if (sdist>variables[DistRVar]->real()) sdist=variables[DistRVar]->real();
	  variables[SDistRVar]->Set(sdist);
      }
      break;
   case PickSliderD:
      {
	  Real sdist = variables[SDistDVar]->real()+dist/2.0;
	  if (sdist<0.0) sdist=0.0;
	  else if (sdist>variables[DistDVar]->real()) sdist=variables[DistDVar]->real();
	  variables[SDistDVar]->Set(sdist);
      }
      break;
   case PickCyls:
      MoveDelta(delta);
      break;
   default:
     break;
   }
   execute(0);
}


/***************************************************************************
 * This standard method simply moves all the widget's PointVariables by
 *      the same delta.
 * The last line of this method should call the BaseWidget execute method
 *      (which calls the redraw method).
 */
void
ScaledFrameWidget::MoveDelta( const Vector& delta )
{
   variables[PointRVar]->MoveDelta(delta);
   variables[PointDVar]->MoveDelta(delta);
   variables[CenterVar]->MoveDelta(delta);

   execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
ScaledFrameWidget::ReferencePoint() const
{
   return variables[CenterVar]->point();
}


void
ScaledFrameWidget::SetPosition( const Point& center, const Point& R, const Point& D )
{
   variables[PointRVar]->Move(R);
   variables[PointDVar]->Move(D);
   Real sizeR((R-center).length());
   Real sizeD((D-center).length());
   variables[DistRVar]->Move(sizeR);
   variables[CenterVar]->Move(center);
   variables[DistDVar]->Set(sizeD);
//   variables[DistDVar]->Move(sizeD);
//   variables[CenterVar]->Set(center, Scheme3); // This should set Hypo...
   variables[SDistRVar]->Set(sizeR*variables[RatioRVar]->real(), Scheme1); // Slider1...
   variables[SDistDVar]->Set(sizeD*variables[RatioDVar]->real(), Scheme1); // Slider2...

   execute(0);
}


void
ScaledFrameWidget::GetPosition( Point& center, Point& R, Point& D )
{
   center = variables[CenterVar]->point();
   R = variables[PointRVar]->point();
   D = variables[PointDVar]->point();
}


void
ScaledFrameWidget::SetPosition( const Point& center, const Vector& normal,
				const Real size1, const Real size2 )
{
   Vector axis1, axis2;
   normal.find_orthogonal(axis1, axis2);
   
   variables[PointRVar]->Move(center+axis1*size1);
   variables[PointDVar]->Move(center+axis2*size2);
   variables[CenterVar]->Move(center);
   variables[DistRVar]->Move(size1);
   variables[DistDVar]->Set(size2); // This should set the Hypo...
   variables[SDistRVar]->Set(size1*variables[RatioRVar]->real(), Scheme1); // Slider1...
   variables[SDistDVar]->Set(size2*variables[RatioDVar]->real(), Scheme1); // Slider2...

   execute(0);
}


void
ScaledFrameWidget::GetPosition( Point& center, Vector& normal,
				Real& size1, Real& size2 )
{
   center = variables[CenterVar]->point();
   normal = Cross(GetRightAxis(), GetDownAxis());
   size1 = variables[DistRVar]->real();
   size2 = variables[DistDVar]->real();
}


void
ScaledFrameWidget::SetRatioR( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[RatioRVar]->Set(ratio);
   
   execute(0);
}


Real
ScaledFrameWidget::GetRatioR() const
{
   return (variables[RatioRVar]->real());
}


void
ScaledFrameWidget::SetRatioD( const Real ratio )
{
   ASSERT((ratio>=0.0) && (ratio<=1.0));
   variables[RatioDVar]->Set(ratio);
   
   execute(0);
}


Real
ScaledFrameWidget::GetRatioD() const
{
   return (variables[RatioDVar]->real());
}


void
ScaledFrameWidget::SetSize( const Real sizeR, const Real sizeD )
{
   ASSERT((sizeR>=0.0)&&(sizeD>=0.0));

   Point center(variables[CenterVar]->point());
   Vector axisR(variables[PointRVar]->point() - center);
   Vector axisD(variables[PointDVar]->point() - center);
   Real ratioR(sizeR/variables[DistRVar]->real());
   Real ratioD(sizeD/variables[DistDVar]->real());

   variables[PointRVar]->Move(center+axisR*ratioR);
   variables[PointDVar]->Move(center+axisD*ratioD);

   variables[DistRVar]->Move(sizeR);
   variables[DistDVar]->Set(sizeD); // This should set the Hypo...
   variables[SDistRVar]->Set(sizeR*variables[RatioRVar]->real(), Scheme1); // Slider1...
   variables[SDistDVar]->Set(sizeD*variables[RatioDVar]->real(), Scheme1); // Slider2...

   execute(0);
}

void
ScaledFrameWidget::GetSize( Real& sizeR, Real& sizeD ) const
{
   sizeR = variables[DistRVar]->real();
   sizeD = variables[DistDVar]->real();
}


const Vector&
ScaledFrameWidget::GetRightAxis()
{
   Vector axis(variables[PointRVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldrightaxis;
   else
      return (oldrightaxis = axis.normal());
}


const Vector&
ScaledFrameWidget::GetDownAxis()
{
   Vector axis(variables[PointDVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return olddownaxis;
   else
      return (olddownaxis = axis.normal());
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
clString
ScaledFrameWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   
   switch(mindex){
   case 0:
      return "Point";
   case 1:
      return "Edge";
   case 2:
      return "Resize";
   case 3:
      return "Slider";
   default:
      return "UnknownMaterial";
   }
}

} // End namespace Widgets
} // End namespace PSECore

//
// $Log$
// Revision 1.2.2.3  2000/10/26 14:16:57  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.4  2000/06/26 21:54:19  bigler
// Added new modes to allow for more control over which pieces
// are rendered.
//
// Revision 1.3  2000/06/21 20:57:25  bigler
// Added additional modes for widget grid.
// One of the additional modes create a scaled widget frame that allows for resizing, but restricts movement to be axis alligned (no rotation).
// The other mode allows for rotation without resizing.
//
// Revision 1.2  1999/08/17 06:38:32  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:09  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//
