
/*
 *  FrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/FrameWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Constraints/PlaneConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 8;
const Index NumVars = 7;
const Index NumGeoms = 16;
const Index NumPcks = 9;
const Index NumMdes = 4;
const Index NumSwtchs = 3;
const Index NumSchemes = 6;

enum { ConstULDR, ConstURDL, ConstPyth, ConstPlane,
       ConstULUR, ConstULDL, ConstDRUR, ConstDRDL };
enum { GeomSPointUL, GeomSPointUR, GeomSPointDR, GeomSPointDL,
       GeomPointUL, GeomPointUR, GeomPointDR, GeomPointDL,
       GeomCylU, GeomCylR, GeomCylD, GeomCylL,
       GeomResizeU, GeomResizeR, GeomResizeD, GeomResizeL };
enum { PickSphUL, PickSphUR, PickSphDR, PickSphDL, PickCyls,
       PickResizeU, PickResizeR, PickResizeD, PickResizeL };

FrameWidget::FrameWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumPcks, NumMdes, NumSwtchs, widget_scale),
  oldaxis1(1, 0, 0), oldaxis2(0, 1, 0)
{
   Real INIT = widget_scale;
   // Schemes 5/6 are used by the picks in GeomMoved!!
   variables[PointULVar] = new PointVariable("PntUL", solve, Scheme1, Point(0, 0, 0));
   variables[PointURVar] = new PointVariable("PntUR", solve, Scheme2, Point(INIT, 0, 0));
   variables[PointDRVar] = new PointVariable("PntDR", solve, Scheme3, Point(INIT, INIT, 0));
   variables[PointDLVar] = new PointVariable("PntDL", solve, Scheme4, Point(0, INIT, 0));
   variables[Dist1Var] = new RealVariable("DIST1", solve, Scheme6, INIT);
   variables[Dist2Var] = new RealVariable("DIST2", solve, Scheme5, INIT);
   variables[HypoVar] = new RealVariable("HYPO", solve, Scheme5, sqrt(2*INIT*INIT));

   constraints[ConstPlane] = new PlaneConstraint("ConstPlane",
						 NumSchemes,
						 variables[PointULVar],
						 variables[PointURVar],
						 variables[PointDRVar],
						 variables[PointDLVar]);
   constraints[ConstPlane]->VarChoices(Scheme1, 2, 3, 2, 1);
   constraints[ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 3);
   constraints[ConstPlane]->VarChoices(Scheme3, 0, 3, 0, 1);
   constraints[ConstPlane]->VarChoices(Scheme4, 2, 1, 0, 1);
   constraints[ConstPlane]->VarChoices(Scheme5, 2, 3, 0, 1);
   constraints[ConstPlane]->VarChoices(Scheme6, 2, 3, 0, 1);
   constraints[ConstPlane]->Priorities(P_Highest, P_Highest,
				       P_Highest, P_Highest);
   constraints[ConstULDR] = new DistanceConstraint("Const13",
						   NumSchemes,
						   variables[PointULVar],
						   variables[PointDRVar],
						   variables[HypoVar]);
   constraints[ConstULDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstULDR]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstULDR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstULDR]->VarChoices(Scheme5, 2, 2, 1);
   constraints[ConstULDR]->VarChoices(Scheme6, 1, 0, 1);
   constraints[ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[ConstURDL] = new DistanceConstraint("Const24",
						   NumSchemes,
						   variables[PointURVar],
						   variables[PointDLVar],
						   variables[HypoVar]);
   constraints[ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstURDL]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstURDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstURDL]->VarChoices(Scheme5, 1, 0, 1);
   constraints[ConstURDL]->VarChoices(Scheme6, 2, 2, 1);
   constraints[ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[ConstPyth] = new PythagorasConstraint("ConstPyth",
						     NumSchemes,
						     variables[Dist1Var],
						     variables[Dist2Var],
						     variables[HypoVar]);
   constraints[ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPyth]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ConstPyth]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstPyth]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstPyth]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstPyth]->VarChoices(Scheme6, 2, 2, 1);
   constraints[ConstPyth]->Priorities(P_Default, P_Default, P_HighMedium);
   constraints[ConstULUR] = new DistanceConstraint("Const12",
						   NumSchemes,
						   variables[PointULVar],
						   variables[PointURVar],
						   variables[Dist1Var]);
   constraints[ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstULUR]->VarChoices(Scheme3, 1, 0, 0);
   constraints[ConstULUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ConstULUR]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstULUR]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstULDL] = new DistanceConstraint("Const14",
						   NumSchemes,
						   variables[PointULVar],
						   variables[PointDLVar],
						   variables[Dist2Var]);
   constraints[ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstULDL]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstULDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ConstULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstULDL]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstULDL]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstDRUR] = new DistanceConstraint("Const32",
						   NumSchemes,
						   variables[PointDRVar],
						   variables[PointURVar],
						   variables[Dist2Var]);
   constraints[ConstDRUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstDRUR]->VarChoices(Scheme4, 1, 0, 0);
   constraints[ConstDRUR]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstDRUR]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ConstDRDL] = new DistanceConstraint("Const34",
						   NumSchemes,
						   variables[PointDRVar],
						   variables[PointDLVar],
						   variables[Dist1Var]);
   constraints[ConstDRDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstDRDL]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstDRDL]->VarChoices(Scheme5, 1, 1, 1);
   constraints[ConstDRDL]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   Index geom, pick;
   GeomGroup* cyls = new GeomGroup;
   for (geom = GeomSPointUL; geom <= GeomSPointDL; geom++) {
      geometries[geom] = new GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = GeomCylU; geom <= GeomCylL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[PickCyls] = new GeomPick(cyls, module, this, PickCyls);
   picks[PickCyls]->set_highlight(HighlightMaterial);
   GeomMaterial* cylsm = new GeomMaterial(picks[PickCyls], EdgeMaterial);
   CreateModeSwitch(0, cylsm);

   GeomGroup* pts = new GeomGroup;
   for (geom = GeomPointUL, pick = PickSphUL;
	geom <= GeomPointDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(HighlightMaterial);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, PointMaterial);
   CreateModeSwitch(1, ptsm);
   
   GeomGroup* resizes = new GeomGroup;
   for (geom = GeomResizeU, pick = PickResizeU;
	geom <= GeomResizeL; geom++, pick++) {
      geometries[geom] = new GeomCappedCylinder;
      picks[pick] = new GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(HighlightMaterial);
      resizes->add(picks[pick]);
   }
   GeomMaterial* resizem = new GeomMaterial(resizes, ResizeMaterial);
   CreateModeSwitch(2, resizem);

   SetMode(Mode0, Switch0|Switch1|Switch2);
   SetMode(Mode1, Switch0|Switch1);
   SetMode(Mode2, Switch0|Switch2);
   SetMode(Mode3, Switch0);

   FinishWidget();
}


FrameWidget::~FrameWidget()
{
}


void
FrameWidget::widget_execute()
{
   Real spherediam(widget_scale), resizediam(0.75*widget_scale), cylinderdiam(0.5*widget_scale);
   Point UL(variables[PointULVar]->point());
   Point UR(variables[PointURVar]->point());
   Point DR(variables[PointDRVar]->point());
   Point DL(variables[PointDLVar]->point());

   if (mode_switches[0]->get_state()) {
      ((GeomCylinder*)geometries[GeomCylU])->move(UL, UR, cylinderdiam);
      ((GeomCylinder*)geometries[GeomCylR])->move(UR, DR, cylinderdiam);
      ((GeomCylinder*)geometries[GeomCylD])->move(DR, DL, cylinderdiam);
      ((GeomCylinder*)geometries[GeomCylL])->move(DL, UL, cylinderdiam);
      ((GeomSphere*)geometries[GeomSPointUL])->move(UL, cylinderdiam);
      ((GeomSphere*)geometries[GeomSPointUR])->move(UR, cylinderdiam);
      ((GeomSphere*)geometries[GeomSPointDR])->move(DR, cylinderdiam);
      ((GeomSphere*)geometries[GeomSPointDL])->move(DL, cylinderdiam);
   }
   
   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[GeomPointUL])->move(UL, spherediam);
      ((GeomSphere*)geometries[GeomPointUR])->move(UR, spherediam);
      ((GeomSphere*)geometries[GeomPointDR])->move(DR, spherediam);
      ((GeomSphere*)geometries[GeomPointDL])->move(DL, spherediam);
   }

   if (mode_switches[2]->get_state()) {
      Vector resizelen1(GetAxis1()*0.6*widget_scale),
	 resizelen2(GetAxis2()*0.6*widget_scale);
      
      Point p(UL + (UR - UL) / 2.0);
      ((GeomCappedCylinder*)geometries[GeomResizeU])->move(p - resizelen2, p + resizelen2, resizediam);
      p = UR + (DR - UR) / 2.0;
      ((GeomCappedCylinder*)geometries[GeomResizeR])->move(p - resizelen1, p + resizelen1, resizediam);
      p = DR + (DL - DR) / 2.0;
      ((GeomCappedCylinder*)geometries[GeomResizeD])->move(p - resizelen2, p + resizelen2, resizediam);
      p = DL + (UL - DL) / 2.0;
      ((GeomCappedCylinder*)geometries[GeomResizeL])->move(p - resizelen1, p + resizelen1, resizediam);
   }

   ((DistanceConstraint*)constraints[ConstULUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ConstDRDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ConstULDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ConstDRUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ConstULDR])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);
   ((DistanceConstraint*)constraints[ConstURDL])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);

   Vector spvec1(UR - UL);
   Vector spvec2(DL - UL);
   if ((spvec1.length2() > 1e-6) && (spvec2.length2() > 1e-6)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v(Cross(spvec1, spvec2));
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if ((geom == PickResizeU) || (geom == PickResizeD))
	    picks[geom]->set_principal(spvec2);
	 else if ((geom == PickResizeL) || (geom == PickResizeR))
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
FrameWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 int cbdata )
{
   Vector delt(delta);
   ((DistanceConstraint*)constraints[ConstULUR])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[ConstDRDL])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[ConstULDL])->SetDefault(GetAxis2());
   ((DistanceConstraint*)constraints[ConstDRUR])->SetDefault(GetAxis2());
   
   switch(cbdata){
   case PickSphUL:
      variables[PointULVar]->SetDelta(delta);
      break;
   case PickSphUR:
      variables[PointURVar]->SetDelta(delta);
      break;
   case PickSphDR:
      variables[PointDRVar]->SetDelta(delta);
      break;
   case PickSphDL:
      variables[PointDLVar]->SetDelta(delta);
      break;
   case PickResizeU:
      if (((variables[PointULVar]->point()+delta)-variables[PointDLVar]->point()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[PointDLVar]->point() + delta.normal()*3.2*widget_scale)
		 - variables[PointULVar]->point());
      }
      variables[PointULVar]->MoveDelta(delt);
      variables[PointURVar]->SetDelta(delt, Scheme6);
      break;
   case PickResizeR:
      if (((variables[PointURVar]->point()+delta)-variables[PointULVar]->point()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[PointULVar]->point() + delta.normal()*3.2*widget_scale)
		 - variables[PointURVar]->point());
      }
      variables[PointURVar]->MoveDelta(delt);
      variables[PointDRVar]->SetDelta(delt, Scheme5);
      break;
   case PickResizeD:
      if (((variables[PointDRVar]->point()+delta)-variables[PointURVar]->point()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[PointURVar]->point() + delta.normal()*3.2*widget_scale)
		 - variables[PointDRVar]->point());
      }
      variables[PointDRVar]->MoveDelta(delt);
      variables[PointDLVar]->SetDelta(delt, Scheme6);
      break;
   case PickResizeL:
      if (((variables[PointDLVar]->point()+delta)-variables[PointDRVar]->point()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[PointDRVar]->point() + delta.normal()*3.2*widget_scale)
		 - variables[PointDLVar]->point());
      }
      variables[PointDLVar]->MoveDelta(delt);
      variables[PointULVar]->SetDelta(delt, Scheme5);
      break;
   case PickCyls:
      MoveDelta(delta);
      break;
   }
   execute();
}


void
FrameWidget::MoveDelta( const Vector& delta )
{
   variables[PointULVar]->MoveDelta(delta);
   variables[PointURVar]->MoveDelta(delta);
   variables[PointDRVar]->MoveDelta(delta);
   variables[PointDLVar]->MoveDelta(delta);

   execute();
}


Point
FrameWidget::ReferencePoint() const
{
   return (variables[PointULVar]->point()
	   + (variables[PointDRVar]->point()
	      -variables[PointULVar]->point())/2.0);
}


void
FrameWidget::SetPosition( const Point& UL, const Point& UR, const Point& DL )
{
   variables[PointULVar]->Move(UL);
   variables[PointURVar]->Move(UR);
   variables[PointDLVar]->Move(DL);
   variables[Dist1Var]->Move((UR-UL).length());
   variables[Dist2Var]->Move((DL-UL).length());
   variables[PointDRVar]->Set(UR+(DL-UL), Scheme5); // This should set Hypo...

   execute();
}


void
FrameWidget::GetPosition( Point& UL, Point& UR, Point& DL )
{
   UL = variables[PointULVar]->point();
   UR = variables[PointURVar]->point();
   DL = variables[PointDLVar]->point();
}


void
FrameWidget::SetPosition( const Point& center, const Vector& normal,
			  const Real size1, const Real size2 )
{
   Real s1(size1/2.0), s2(size2/2.0);
   Vector axis1, axis2;
   normal.find_orthogonal(axis1, axis2);
   
   variables[PointULVar]->Move(center-axis1*s1-axis2*s2);
   variables[PointDRVar]->Move(center+axis1*s1+axis2*s2);
   variables[PointURVar]->Move(center+axis1*s1-axis2*s2);
   variables[PointDLVar]->Move(center-axis1*s1+axis2*s2);
   variables[Dist1Var]->Move(size1);
   variables[Dist2Var]->Set(size2); // This should set the Hypo...

   execute();
}


void
FrameWidget::GetPosition( Point& center, Vector& normal,
			  Real& size1, Real& size2 )
{
   center = (variables[PointDRVar]->point()
	     + ((variables[PointULVar]->point()-variables[PointDRVar]->point())
		/ 2.0));
   normal = Cross(GetAxis1(), GetAxis2());
   size1 = variables[Dist1Var]->real();
   size2 = variables[Dist2Var]->real();
}


void
FrameWidget::SetSize( const Real size1, const Real size2 )
{
   ASSERT((size1>=0.0)&&(size1>=0.0));

   Point center(variables[PointDRVar]->point()
		+ ((variables[PointULVar]->point()-variables[PointDRVar]->point())
		   / 2.0));
   Vector axis1((variables[PointURVar]->point() - variables[PointULVar]->point())/2.0);
   Vector axis2((variables[PointDLVar]->point() - variables[PointULVar]->point())/2.0);
   Real ratio1(size1/variables[Dist1Var]->real());
   Real ratio2(size2/variables[Dist2Var]->real());

   variables[PointULVar]->Move(center-axis1*ratio1-axis2*ratio2);
   variables[PointDRVar]->Move(center+axis1*ratio1+axis2*ratio2);
   variables[PointURVar]->Move(center+axis1*ratio1-axis2*ratio2);
   variables[PointDLVar]->Move(center-axis1*ratio1+axis2*ratio2);

   variables[Dist1Var]->Move(size1);
   variables[Dist2Var]->Set(size2); // This should set the Hypo...

   execute();
}

void
FrameWidget::GetSize( Real& size1, Real& size2 ) const
{
   size1 = variables[Dist1Var]->real();
   size2 = variables[Dist2Var]->real();
}


Vector
FrameWidget::GetAxis1()
{
   Vector axis(variables[PointURVar]->point() - variables[PointULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


Vector
FrameWidget::GetAxis2()
{
   Vector axis(variables[PointDLVar]->point() - variables[PointULVar]->point());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


