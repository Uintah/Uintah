/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  ViewWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Dataflow/Widgets/ViewWidget.h>
#include <Dataflow/Constraints/DistanceConstraint.h>
#include <Dataflow/Constraints/LineConstraint.h>
#include <Dataflow/Constraints/ProjectConstraint.h>
#include <Dataflow/Constraints/RatioConstraint.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <math.h>

namespace SCIRun {




const Index NumCons = 5;
const Index NumVars = 7;
const Index NumGeoms = 28;
const Index NumPcks = 8;
const Index NumMatls = 4;
const Index NumMdes = 3;
const Index NumSwtchs = 2;
const Index NumSchemes = 5;

enum { ConstProject, ConstSegment, ConstUpDist, ConstEyeDist, ConstFOV };
enum { GeomPointUL, GeomPointUR, GeomPointDR, GeomPointDL,
       GeomCylU, GeomCylR, GeomCylD, GeomCylL,
       GeomResizeUp, GeomResizeEye, GeomUpVector, GeomUp, 
       GeomEye, GeomFore, GeomLookAt, GeomShaft,
       GeomCornerUL, GeomCornerUR, GeomCornerDR, GeomCornerDL,
       GeomEdgeU, GeomEdgeR, GeomEdgeD, GeomEdgeL,
       GeomDiagUL, GeomDiagUR, GeomDiagDR, GeomDiagDL };
enum { PickUp, PickCyls, PickShaft,
       PickResizeUp, PickResizeEye, PickEye, PickFore, PickLookAt };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
ViewWidget::ViewWidget( Module* module, CrowdMonitor* lock,
			double widget_scale, double AspectRatio )
: BaseWidget(module, lock, "ViewWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  ratio(AspectRatio), oldaxis1(1, 0, 0), oldaxis2(1, 0, 0)
{
   double INIT = 10.0*widget_scale;
   // Scheme5 are used by the pick in GeomMoved!!
   variables[EyeVar] = scinew PointVariable("Eye", solve, Scheme1, Point(0, 0, -5*INIT));
   variables[ForeVar] = scinew PointVariable("Fore", solve, Scheme2, Point(0, 0, -3*INIT));
   variables[UpVar] = scinew PointVariable("UpVar", solve, Scheme3, Point(0, INIT, -3*INIT));
   variables[LookAtVar] = scinew PointVariable("LookAt", solve, Scheme4, Point(0, 0, 0));
   variables[UpDistVar] = scinew RealVariable("UpDist", solve, Scheme5, INIT);
   variables[EyeDistVar] = scinew RealVariable("EyeDist", solve, Scheme1, 2*INIT);
   variables[FOVVar] = scinew RealVariable("FOVDist", solve, Scheme1, 0.5);

   constraints[ConstProject] = scinew ProjectConstraint("Project",
						     NumSchemes,
						     variables[ForeVar],
						     variables[UpVar],
						     variables[EyeVar],
						     variables[LookAtVar]);
   constraints[ConstProject]->VarChoices(Scheme1, 1, 1, 1, 1);
   constraints[ConstProject]->VarChoices(Scheme2, 1, 1, 1, 1);
   constraints[ConstProject]->VarChoices(Scheme3, 1, 1, 1, 1);
   constraints[ConstProject]->VarChoices(Scheme4, 1, 1, 1, 1);
   constraints[ConstProject]->VarChoices(Scheme5, 1, 1, 1, 1);
   constraints[ConstProject]->Priorities(P_Lowest, P_Lowest, P_Lowest);
   constraints[ConstSegment] = scinew LineConstraint("Segment",
						     NumSchemes,
						     variables[EyeVar],
						     variables[LookAtVar],
						     variables[ForeVar]);
   constraints[ConstSegment]->VarChoices(Scheme1, 2, 2, 2);
   constraints[ConstSegment]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstSegment]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstSegment]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstSegment]->VarChoices(Scheme5, 2, 2, 2);
   constraints[ConstSegment]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstUpDist] = scinew DistanceConstraint("UpDist",
						     NumSchemes,
						     variables[UpVar],
						     variables[ForeVar],
						     variables[UpDistVar]);
   constraints[ConstUpDist]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstUpDist]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstUpDist]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstUpDist]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstUpDist]->VarChoices(Scheme5, 2, 2, 2);
   constraints[ConstUpDist]->Priorities(P_HighMedium, P_HighMedium, P_HighMedium);
   constraints[ConstEyeDist] = scinew DistanceConstraint("EyeDist",
						     NumSchemes,
						     variables[EyeVar],
						     variables[ForeVar],
						     variables[EyeDistVar]);
   constraints[ConstEyeDist]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstEyeDist]->VarChoices(Scheme2, 2, 2, 2);
   constraints[ConstEyeDist]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstEyeDist]->VarChoices(Scheme4, 1, 1, 1);
   constraints[ConstEyeDist]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstEyeDist]->Priorities(P_HighMedium, P_HighMedium, P_HighMedium);
   constraints[ConstFOV] = scinew RatioConstraint("FOV",
					       NumSchemes,
					       variables[UpDistVar],
					       variables[EyeDistVar],
					       variables[FOVVar]);
   constraints[ConstFOV]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstFOV]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstFOV]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstFOV]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstFOV]->VarChoices(Scheme5, 2, 2, 2);
   constraints[ConstFOV]->Priorities(P_Default, P_Default, P_Default);

   GeomGroup* eyes = scinew GeomGroup;
   geometries[GeomEye] = scinew GeomSphere;
   picks[PickEye] = scinew GeomPick(geometries[GeomEye], module, this, PickEye);
   picks[PickEye]->set_highlight(DefaultHighlightMaterial);
   eyes->add(picks[PickEye]);

   geometries[GeomUp] = scinew GeomSphere;
   picks[PickUp] = scinew GeomPick(geometries[GeomUp], module, this, PickUp);
   picks[PickUp]->set_highlight(DefaultHighlightMaterial);
   eyes->add(picks[PickUp]);

   geometries[GeomFore] = scinew GeomSphere;
   picks[PickFore] = scinew GeomPick(geometries[GeomFore], module, this, PickFore);
   picks[PickFore]->set_highlight(DefaultHighlightMaterial);
   eyes->add(picks[PickFore]);

   geometries[GeomLookAt] = scinew GeomSphere;
   picks[PickLookAt] = scinew GeomPick(geometries[GeomLookAt], module, this, PickLookAt);
   picks[PickLookAt]->set_highlight(DefaultHighlightMaterial);
   eyes->add(picks[PickLookAt]);
   materials[EyesMatl] = scinew GeomMaterial(eyes, DefaultPointMaterial);

   Index geom;
   GeomGroup* resizes = scinew GeomGroup;
   geometries[GeomResizeUp] = scinew GeomCappedCylinder;
   picks[PickResizeUp] = scinew GeomPick(geometries[GeomResizeUp], module, this, PickResizeUp);
   picks[PickResizeUp]->set_highlight(DefaultHighlightMaterial);
   resizes->add(picks[PickResizeUp]);
   geometries[GeomResizeEye] = scinew GeomCappedCylinder;
   picks[PickResizeEye] = scinew GeomPick(geometries[GeomResizeEye], module,
				       this, PickResizeEye);
   picks[PickResizeEye]->set_highlight(DefaultHighlightMaterial);
   resizes->add(picks[PickResizeEye]);
   materials[ResizeMatl] = scinew GeomMaterial(resizes, DefaultResizeMaterial);

   GeomGroup* shafts = scinew GeomGroup;
   geometries[GeomUpVector] = scinew GeomCylinder;
   shafts->add(geometries[GeomUpVector]);
   geometries[GeomShaft] = scinew GeomCylinder;
   shafts->add(geometries[GeomShaft]);
   picks[PickShaft] = scinew GeomPick(shafts, module, this, PickShaft);
   picks[PickShaft]->set_highlight(DefaultHighlightMaterial);
   materials[ShaftMatl] = scinew GeomMaterial(picks[PickShaft], DefaultEdgeMaterial);

   GeomGroup* w = scinew GeomGroup;
   w->add(materials[EyesMatl]);
   w->add(materials[ResizeMatl]);
   w->add(materials[ShaftMatl]);
   CreateModeSwitch(0, w);

   GeomGroup* cyls = scinew GeomGroup;
   for (geom = GeomPointUL; geom <= GeomPointDL; geom++) {
      geometries[geom] = scinew GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomCylU; geom <= GeomCylL; geom++) {
      geometries[geom] = scinew GeomCylinder;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomCornerUL; geom <= GeomCornerDL; geom++) {
      geometries[geom] = scinew GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomEdgeU; geom <= GeomEdgeL; geom++) {
      geometries[geom] = scinew GeomCylinder;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomDiagUL; geom <= GeomDiagDL; geom++) {
      geometries[geom] = scinew GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[PickCyls] = scinew GeomPick(cyls, module, this, PickCyls);
   picks[PickCyls]->set_highlight(DefaultHighlightMaterial);
   materials[FrustrumMatl] = scinew GeomMaterial(picks[PickCyls], DefaultEdgeMaterial);
   CreateModeSwitch(1, materials[FrustrumMatl]);

   SetMode(Mode0, Switch0|Switch1);
   SetMode(Mode1, Switch0);
   SetMode(Mode2, Switch1);

   FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
ViewWidget::~ViewWidget()
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
ViewWidget::redraw()
{
   double sphererad(widget_scale), cylinderrad(0.5*widget_scale);

   if (mode_switches[0]->get_state()) {
      ((GeomSphere*)geometries[GeomEye])->move(variables[EyeVar]->point(), sphererad);
      ((GeomSphere*)geometries[GeomFore])->move(variables[ForeVar]->point(), sphererad);
      ((GeomSphere*)geometries[GeomUp])->move(variables[UpVar]->point(), sphererad);
      ((GeomSphere*)geometries[GeomLookAt])->move(variables[LookAtVar]->point(), sphererad);
      ((GeomCylinder*)geometries[GeomShaft])->move(variables[EyeVar]->point(),
						   variables[LookAtVar]->point(),
						   cylinderrad);
      ((GeomCylinder*)geometries[GeomUpVector])->move(variables[UpVar]->point(),
						      variables[ForeVar]->point(),
						      cylinderrad);
      ((GeomCappedCylinder*)geometries[GeomResizeUp])->move(variables[UpVar]->point(),
							    variables[UpVar]->point()
							    + (GetUpAxis() * 1.5 * widget_scale),
							    cylinderrad);
      ((GeomCappedCylinder*)geometries[GeomResizeEye])->move(variables[EyeVar]->point(),
							     variables[EyeVar]->point()
							     - (GetEyeAxis() * 1.5 * widget_scale),
							     cylinderrad);
   }

   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[GeomPointUL])->move(GetFrontUL(), cylinderrad);
      ((GeomSphere*)geometries[GeomPointUR])->move(GetFrontUR(), cylinderrad);
      ((GeomSphere*)geometries[GeomPointDR])->move(GetFrontDR(), cylinderrad);
      ((GeomSphere*)geometries[GeomPointDL])->move(GetFrontDL(), cylinderrad);
      ((GeomSphere*)geometries[GeomCornerUL])->move(GetBackUL(), cylinderrad);
      ((GeomSphere*)geometries[GeomCornerUR])->move(GetBackUR(), cylinderrad);
      ((GeomSphere*)geometries[GeomCornerDR])->move(GetBackDR(), cylinderrad);
      ((GeomSphere*)geometries[GeomCornerDL])->move(GetBackDL(), cylinderrad);
      ((GeomCylinder*)geometries[GeomCylU])->move(GetFrontUL(), GetFrontUR(), cylinderrad);
      ((GeomCylinder*)geometries[GeomCylR])->move(GetFrontUR(), GetFrontDR(), cylinderrad);
      ((GeomCylinder*)geometries[GeomCylD])->move(GetFrontDR(), GetFrontDL(), cylinderrad);
      ((GeomCylinder*)geometries[GeomCylL])->move(GetFrontDL(), GetFrontUL(), cylinderrad);
      ((GeomCylinder*)geometries[GeomEdgeU])->move(GetBackUL(), GetBackUR(), cylinderrad);
      ((GeomCylinder*)geometries[GeomEdgeR])->move(GetBackUR(), GetBackDR(), cylinderrad);
      ((GeomCylinder*)geometries[GeomEdgeD])->move(GetBackDR(), GetBackDL(), cylinderrad);
      ((GeomCylinder*)geometries[GeomEdgeL])->move(GetBackDL(), GetBackUL(), cylinderrad);
      ((GeomCylinder*)geometries[GeomDiagUL])->move(GetFrontUL(), GetBackUL(), cylinderrad);
      ((GeomCylinder*)geometries[GeomDiagUR])->move(GetFrontUR(), GetBackUR(), cylinderrad);
      ((GeomCylinder*)geometries[GeomDiagDR])->move(GetFrontDR(), GetBackDR(), cylinderrad);
      ((GeomCylinder*)geometries[GeomDiagDL])->move(GetFrontDL(), GetBackDL(), cylinderrad);
   }

   Vector spvec1(GetEyeAxis());
   Vector spvec2(GetUpAxis());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if (geom == PickResizeUp)
	    picks[geom]->set_principal(spvec2);
	 if ((geom == PickResizeEye) || (geom == PickFore))
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
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
ViewWidget::geom_moved( GeomPick*, int /* axis */, double /* dist */,
			const Vector& delta, int pick, const BState& )
{
   switch(pick){
   case PickEye:
      variables[EyeVar]->SetDelta(delta);
      break;
   case PickFore:
      variables[ForeVar]->SetDelta(delta);
      break;
   case PickLookAt:
      variables[LookAtVar]->SetDelta(delta);
      break;
   case PickUp:
      variables[UpVar]->SetDelta(delta);
      break;
   case PickResizeUp:
      variables[UpVar]->SetDelta(delta, Scheme5);
      break;
   case PickResizeEye:
      variables[EyeVar]->SetDelta(delta);
      break;
   case PickCyls:
   case PickShaft:
      MoveDelta(delta);
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
ViewWidget::MoveDelta( const Vector& delta )
{
   variables[EyeVar]->MoveDelta(delta);
   variables[ForeVar]->MoveDelta(delta);
   variables[UpVar]->MoveDelta(delta);
   variables[LookAtVar]->MoveDelta(delta);

   execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
ViewWidget::ReferencePoint() const
{
   return variables[EyeVar]->point();
}


View
ViewWidget::GetView()
{
   return View(variables[EyeVar]->point(), variables[LookAtVar]->point(),
	       GetUpVector(), GetFOV());
}


Vector
ViewWidget::GetUpVector()
{
   return GetUpAxis();
}


double
ViewWidget::GetFOV() const
{
   return 2.0*atan(variables[FOVVar]->real());
}


void
ViewWidget::SetView( const View& view )
{
   variables[EyeVar]->Move(view.eyep());
   variables[LookAtVar]->Move(view.lookat());
   variables[ForeVar]->Move(view.eyep()+(view.eyep()-view.lookat())/2.0);
   variables[EyeDistVar]->Move((view.lookat()-view.eyep()).length()/2.0);
   variables[FOVVar]->Set(tan(view.fov()/2.0)); // Should Set UpVar/UpDistVar...
   
   execute(0);
}


double
ViewWidget::GetAspectRatio() const
{
   return ratio;
}


void
ViewWidget::SetAspectRatio( const double aspect )
{
   ratio = aspect;

   execute(0);
}


const Vector&
ViewWidget::GetEyeAxis()
{
   Vector axis(variables[LookAtVar]->point() - variables[EyeVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


const Vector&
ViewWidget::GetUpAxis()
{
   Vector axis(variables[UpVar]->point() - variables[ForeVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


Point
ViewWidget::GetFrontUL()
{
   return (variables[ForeVar]->point()
	   + GetUpAxis() * variables[UpDistVar]->real()
	   + Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio);
}


Point
ViewWidget::GetFrontUR()
{
   return (variables[ForeVar]->point()
	   + GetUpAxis() * variables[UpDistVar]->real()
	   - Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio);
}


Point
ViewWidget::GetFrontDR()
{
   return (variables[ForeVar]->point()
	   - GetUpAxis() * variables[UpDistVar]->real()
	   - Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio);
}


Point
ViewWidget::GetFrontDL()
{
   return (variables[ForeVar]->point()
	   - GetUpAxis() * variables[UpDistVar]->real()
	   + Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio);
}


Point
ViewWidget::GetBackUL()
{
   double t((variables[LookAtVar]->point()-variables[EyeVar]->point()).length()
	  / variables[EyeDistVar]->real());
   return (variables[LookAtVar]->point()
	   + GetUpAxis() * variables[UpDistVar]->real() * t
	   + Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio * t);
}


Point
ViewWidget::GetBackUR()
{
   double t((variables[LookAtVar]->point()-variables[EyeVar]->point()).length()
	  / variables[EyeDistVar]->real());
   return (variables[LookAtVar]->point()
	   + GetUpAxis() * variables[UpDistVar]->real() * t
	   - Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio * t);
}


Point
ViewWidget::GetBackDR()
{
   double t((variables[LookAtVar]->point()-variables[EyeVar]->point()).length()
	  / variables[EyeDistVar]->real());
   return (variables[LookAtVar]->point()
	   - GetUpAxis() * variables[UpDistVar]->real() * t
	   - Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio * t);
}


Point
ViewWidget::GetBackDL()
{
   double t((variables[LookAtVar]->point()-variables[EyeVar]->point()).length()
	  / variables[EyeDistVar]->real());
   return (variables[LookAtVar]->point()
	   - GetUpAxis() * variables[UpDistVar]->real() * t
	   + Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio * t);
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
ViewWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<materials.size());
   
   switch(mindex){
   case 0:
      return "Eyes";
   case 1:
      return "Resize";
   case 2:
      return "Shaft";
   case 3:
      return "Frustrum";
   default:
      return "UnknownMaterial";
   }
}


} // End namespace SCIRun

