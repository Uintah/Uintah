
/*
 *  ViewWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/ViewWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/LineConstraint.h>
#include <Constraints/ProjectConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 5;
const Index NumVars = 7;
const Index NumGeoms = 28;
const Index NumPcks = 8;
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

ViewWidget::ViewWidget( Module* module, CrowdMonitor* lock, Real widget_scale,
			const Real AspectRatio )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumPcks, NumMdes, NumSwtchs, widget_scale*0.1),
  ratio(AspectRatio), oldaxis1(1, 0, 0), oldaxis2(1, 0, 0)
{
   Real INIT = 1.0*widget_scale;
   // Scheme5 are used by the pick in GeomMoved!!
   variables[EyeVar] = new PointVariable("Eye", solve, Scheme1, Point(0, 0, -5*INIT));
   variables[ForeVar] = new PointVariable("Fore", solve, Scheme2, Point(0, 0, -3*INIT));
   variables[UpVar] = new PointVariable("UpVar", solve, Scheme3, Point(0, INIT, -3*INIT));
   variables[LookAtVar] = new PointVariable("LookAt", solve, Scheme4, Point(0, 0, 0));
   variables[UpDistVar] = new RealVariable("UpDist", solve, Scheme5, INIT);
   variables[EyeDistVar] = new RealVariable("EyeDist", solve, Scheme1, 2*INIT);
   variables[FOVVar] = new RealVariable("FOVDist", solve, Scheme1, 0.5);

   constraints[ConstProject] = new ProjectConstraint("Project",
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
   constraints[ConstSegment] = new LineConstraint("Segment",
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
   constraints[ConstUpDist] = new DistanceConstraint("UpDist",
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
   constraints[ConstEyeDist] = new DistanceConstraint("EyeDist",
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
   constraints[ConstFOV] = new RatioConstraint("FOV",
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

   GeomGroup* eyes = new GeomGroup;
   geometries[GeomEye] = new GeomSphere;
   picks[PickEye] = new GeomPick(geometries[GeomEye], module, this, PickEye);
   picks[PickEye]->set_highlight(HighlightMaterial);
   eyes->add(picks[PickEye]);

   geometries[GeomUp] = new GeomSphere;
   picks[PickUp] = new GeomPick(geometries[GeomUp], module, this, PickUp);
   picks[PickUp]->set_highlight(HighlightMaterial);
   eyes->add(picks[PickUp]);

   geometries[GeomFore] = new GeomSphere;
   picks[PickFore] = new GeomPick(geometries[GeomFore], module, this, PickFore);
   picks[PickFore]->set_highlight(HighlightMaterial);
   eyes->add(picks[PickFore]);

   geometries[GeomLookAt] = new GeomSphere;
   picks[PickLookAt] = new GeomPick(geometries[GeomLookAt], module, this, PickLookAt);
   picks[PickLookAt]->set_highlight(HighlightMaterial);
   eyes->add(picks[PickLookAt]);
   GeomMaterial* eyesm = new GeomMaterial(eyes, PointMaterial);

   Index geom;
   GeomGroup* resizes = new GeomGroup;
   geometries[GeomResizeUp] = new GeomCappedCylinder;
   picks[PickResizeUp] = new GeomPick(geometries[GeomResizeUp], module, this, PickResizeUp);
   picks[PickResizeUp]->set_highlight(HighlightMaterial);
   resizes->add(picks[PickResizeUp]);
   geometries[GeomResizeEye] = new GeomCappedCylinder;
   picks[PickResizeEye] = new GeomPick(geometries[GeomResizeEye], module,
				       this, PickResizeEye);
   picks[PickResizeEye]->set_highlight(HighlightMaterial);
   resizes->add(picks[PickResizeEye]);
   GeomMaterial* resizem = new GeomMaterial(resizes, ResizeMaterial);

   GeomGroup* shafts = new GeomGroup;
   geometries[GeomUpVector] = new GeomCylinder;
   shafts->add(geometries[GeomUpVector]);
   geometries[GeomShaft] = new GeomCylinder;
   shafts->add(geometries[GeomShaft]);
   picks[PickShaft] = new GeomPick(shafts, module, this, PickShaft);
   picks[PickShaft]->set_highlight(HighlightMaterial);
   GeomMaterial* shaftm = new GeomMaterial(picks[PickShaft], EdgeMaterial);

   GeomGroup* w = new GeomGroup;
   w->add(eyesm);
   w->add(resizem);
   w->add(shaftm);
   CreateModeSwitch(0, w);

   GeomGroup* cyls = new GeomGroup;
   for (geom = GeomPointUL; geom <= GeomPointDL; geom++) {
      geometries[geom] = new GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomCylU; geom <= GeomCylL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomCornerUL; geom <= GeomCornerDL; geom++) {
      geometries[geom] = new GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomEdgeU; geom <= GeomEdgeL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomDiagUL; geom <= GeomDiagDL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[PickCyls] = new GeomPick(cyls, module, this, PickCyls);
   picks[PickCyls]->set_highlight(HighlightMaterial);
   GeomMaterial* cylsm = new GeomMaterial(picks[PickCyls], EdgeMaterial);
   CreateModeSwitch(1, cylsm);

   SetMode(Mode0, Switch0|Switch1);
   SetMode(Mode1, Switch0);
   SetMode(Mode2, Switch1);

   FinishWidget();
}


ViewWidget::~ViewWidget()
{
}


void
ViewWidget::widget_execute()
{
   Real spherediam(widget_scale), cylinderdiam(0.5*widget_scale);

   if (mode_switches[0]->get_state()) {
      ((GeomSphere*)geometries[GeomEye])->move(variables[EyeVar]->point(), spherediam);
      ((GeomSphere*)geometries[GeomFore])->move(variables[ForeVar]->point(), spherediam);
      ((GeomSphere*)geometries[GeomUp])->move(variables[UpVar]->point(), spherediam);
      ((GeomSphere*)geometries[GeomLookAt])->move(variables[LookAtVar]->point(), spherediam);
      ((GeomCylinder*)geometries[GeomShaft])->move(variables[EyeVar]->point(),
						   variables[LookAtVar]->point(),
						   cylinderdiam);
      ((GeomCylinder*)geometries[GeomUpVector])->move(variables[UpVar]->point(),
						      variables[ForeVar]->point(),
						      cylinderdiam);
      ((GeomCappedCylinder*)geometries[GeomResizeUp])->move(variables[UpVar]->point(),
							    variables[UpVar]->point()
							    + (GetUpAxis() * 1.5 * widget_scale),
							    cylinderdiam);
      ((GeomCappedCylinder*)geometries[GeomResizeEye])->move(variables[EyeVar]->point(),
							     variables[EyeVar]->point()
							     - (GetEyeAxis() * 1.5 * widget_scale),
							     cylinderdiam);
   }

   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[GeomPointUL])->move(GetFrontUL(), cylinderdiam);
      ((GeomSphere*)geometries[GeomPointUR])->move(GetFrontUR(), cylinderdiam);
      ((GeomSphere*)geometries[GeomPointDR])->move(GetFrontDR(), cylinderdiam);
      ((GeomSphere*)geometries[GeomPointDL])->move(GetFrontDL(), cylinderdiam);
      ((GeomSphere*)geometries[GeomCornerUL])->move(GetBackUL(), cylinderdiam);
      ((GeomSphere*)geometries[GeomCornerUR])->move(GetBackUR(), cylinderdiam);
      ((GeomSphere*)geometries[GeomCornerDR])->move(GetBackDR(), cylinderdiam);
      ((GeomSphere*)geometries[GeomCornerDL])->move(GetBackDL(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomCylU])->move(GetFrontUL(), GetFrontUR(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomCylR])->move(GetFrontUR(), GetFrontDR(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomCylD])->move(GetFrontDR(), GetFrontDL(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomCylL])->move(GetFrontDL(), GetFrontUL(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomEdgeU])->move(GetBackUL(), GetBackUR(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomEdgeR])->move(GetBackUR(), GetBackDR(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomEdgeD])->move(GetBackDR(), GetBackDL(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomEdgeL])->move(GetBackDL(), GetBackUL(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomDiagUL])->move(GetFrontUL(), GetBackUL(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomDiagUR])->move(GetFrontUR(), GetBackUR(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomDiagDR])->move(GetFrontDR(), GetBackDR(), cylinderdiam);
      ((GeomCylinder*)geometries[GeomDiagDL])->move(GetFrontDL(), GetBackDL(), cylinderdiam);
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

void
ViewWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
		        int cbdata )
{
   Vector delt(delta);
   
   switch(cbdata){
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
   execute();
}


void
ViewWidget::MoveDelta( const Vector& delta )
{
   variables[EyeVar]->MoveDelta(delta);
   variables[ForeVar]->MoveDelta(delta);
   variables[UpVar]->MoveDelta(delta);
   variables[LookAtVar]->MoveDelta(delta);

   execute();
}


Point
ViewWidget::ReferencePoint() const
{
   return variables[EyeVar]->point();
}


View&
ViewWidget::GetView()
{
   return View(variables[EyeVar]->point(), variables[LookAtVar]->point(),
	       GetUpVector(), GetFOV());
}


Vector&
ViewWidget::GetUpVector()
{
   return GetUpAxis();
}


Real
ViewWidget::GetFOV() const
{
   return 2.0*atan(variables[FOVVar]->real());
}


void
ViewWidget::SetAspectRatio( const Real aspect )
{
   ratio = aspect;

   execute();
}


Real
ViewWidget::GetAspectRatio() const
{
   return ratio;
}


Vector
ViewWidget::GetEyeAxis()
{
   Vector axis(variables[LookAtVar]->point() - variables[EyeVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


Vector
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
   Real t((variables[LookAtVar]->point()-variables[EyeVar]->point()).length()
	  / variables[EyeDistVar]->real());
   return (variables[LookAtVar]->point()
	   + GetUpAxis() * variables[UpDistVar]->real() * t
	   + Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio * t);
}


Point
ViewWidget::GetBackUR()
{
   Real t((variables[LookAtVar]->point()-variables[EyeVar]->point()).length()
	  / variables[EyeDistVar]->real());
   return (variables[LookAtVar]->point()
	   + GetUpAxis() * variables[UpDistVar]->real() * t
	   - Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio * t);
}


Point
ViewWidget::GetBackDR()
{
   Real t((variables[LookAtVar]->point()-variables[EyeVar]->point()).length()
	  / variables[EyeDistVar]->real());
   return (variables[LookAtVar]->point()
	   - GetUpAxis() * variables[UpDistVar]->real() * t
	   - Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio * t);
}


Point
ViewWidget::GetBackDL()
{
   Real t((variables[LookAtVar]->point()-variables[EyeVar]->point()).length()
	  / variables[EyeDistVar]->real());
   return (variables[LookAtVar]->point()
	   - GetUpAxis() * variables[UpDistVar]->real() * t
	   + Cross(GetUpAxis(), GetEyeAxis()) * variables[UpDistVar]->real() * ratio * t);
}


