
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
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 4;
const Index NumVars = 6;
const Index NumGeoms = 16;
const Index NumPcks = 9;
const Index NumMatls = 3;
const Index NumMdes = 2;
const Index NumSwtchs = 3;
const Index NumSchemes = 4;

enum { ConstRD, ConstRC, ConstDC, ConstPyth };
enum { GeomSPointUL, GeomSPointUR, GeomSPointDR, GeomSPointDL,
       GeomPointU, GeomPointR, GeomPointD, GeomPointL,
       GeomCylU, GeomCylR, GeomCylD, GeomCylL,
       GeomResizeU, GeomResizeR, GeomResizeD, GeomResizeL };
enum { PickSphU, PickSphR, PickSphD, PickSphL, PickCyls,
       PickResizeU, PickResizeR, PickResizeD, PickResizeL };

FrameWidget::FrameWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, "FrameWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  oldrightaxis(1, 0, 0), olddownaxis(0, 1, 0)
{
   Real INIT = 5.0*widget_scale;
   // Schemes 5/6 are used by the picks in GeomMoved!!
   variables[CenterVar] = new PointVariable("Center", solve, Scheme1, Point(0, 0, 0));
   variables[PointRVar] = new PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
   variables[PointDVar] = new PointVariable("PntD", solve, Scheme2, Point(0, INIT, 0));
   variables[DistRVar] = new RealVariable("RDIST", solve, Scheme3, INIT);
   variables[DistDVar] = new RealVariable("DDIST", solve, Scheme4, INIT);
   variables[HypoVar] = new RealVariable("HYPO", solve, Scheme3, sqrt(2*INIT*INIT));

   constraints[ConstRD] = new DistanceConstraint("ConstRD",
						 NumSchemes,
						 variables[PointRVar],
						 variables[PointDVar],
						 variables[HypoVar]);
   constraints[ConstRD]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstRD]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRD]->VarChoices(Scheme3, 2, 2, 1);
   constraints[ConstRD]->VarChoices(Scheme4, 2, 2, 0);
   constraints[ConstRD]->Priorities(P_Default, P_Default, P_Default);
   constraints[ConstPyth] = new PythagorasConstraint("ConstPyth",
						     NumSchemes,
						     variables[DistRVar],
						     variables[DistDVar],
						     variables[HypoVar]);
   constraints[ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPyth]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstPyth]->VarChoices(Scheme3, 2, 2, 1);
   constraints[ConstPyth]->VarChoices(Scheme4, 2, 2, 0);
   constraints[ConstPyth]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRC] = new DistanceConstraint("ConstRC",
						 NumSchemes,
						 variables[PointRVar],
						 variables[CenterVar],
						 variables[DistRVar]);
   constraints[ConstRC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme3, 2, 2, 2);
   constraints[ConstRC]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstRC]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstDC] = new DistanceConstraint("ConstDC",
					       NumSchemes,
					       variables[PointDVar],
					       variables[CenterVar],
					       variables[DistDVar]);
   constraints[ConstDC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstDC]->Priorities(P_Highest, P_Highest, P_Default);

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
   picks[PickCyls]->set_highlight(DefaultHighlightMaterial);
   materials[EdgeMatl] = new GeomMaterial(picks[PickCyls], DefaultEdgeMaterial);
   CreateModeSwitch(0, materials[EdgeMatl]);

   GeomGroup* pts = new GeomGroup;
   for (geom = GeomPointU, pick = PickSphU;
	geom <= GeomPointL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(DefaultHighlightMaterial);
      pts->add(picks[pick]);
   }
   materials[PointMatl] = new GeomMaterial(pts, DefaultPointMaterial);
   CreateModeSwitch(1, materials[PointMatl]);
   
   GeomGroup* resizes = new GeomGroup;
   for (geom = GeomResizeU, pick = PickResizeU;
	geom <= GeomResizeL; geom++, pick++) {
      geometries[geom] = new GeomCappedCylinder;
      picks[pick] = new GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(DefaultHighlightMaterial);
      resizes->add(picks[pick]);
   }
   materials[ResizeMatl] = new GeomMaterial(resizes, DefaultResizeMaterial);
   CreateModeSwitch(2, materials[ResizeMatl]);

   SetMode(Mode0, Switch0|Switch1|Switch2);
   SetMode(Mode1, Switch0);

   FinishWidget();
}


FrameWidget::~FrameWidget()
{
}


void
FrameWidget::redraw()
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
   
   if (mode_switches[1]->get_state()) {
      ((GeomSphere*)geometries[GeomPointU])->move(U, sphererad);
      ((GeomSphere*)geometries[GeomPointR])->move(R, sphererad);
      ((GeomSphere*)geometries[GeomPointD])->move(D, sphererad);
      ((GeomSphere*)geometries[GeomPointL])->move(L, sphererad);
   }

   if (mode_switches[2]->get_state()) {
      Vector resizeR(GetRightAxis()*1.5*widget_scale),
	 resizeD(GetDownAxis()*1.5*widget_scale);
      
      ((GeomCappedCylinder*)geometries[GeomResizeU])->move(U, U - resizeD, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeR])->move(R, R + resizeR, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeD])->move(D, D + resizeD, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeL])->move(L, L - resizeR, resizerad);
   }

   ((DistanceConstraint*)constraints[ConstRC])->SetMinimum(1.6*widget_scale);
   ((DistanceConstraint*)constraints[ConstDC])->SetMinimum(1.6*widget_scale);
   ((DistanceConstraint*)constraints[ConstRD])->SetMinimum(sqrt(2*1.6*1.6)*widget_scale);

   Right.normalize();
   Down.normalize();
   Vector Norm(Cross(Right, Down));
   for (Index geom = 0; geom < NumPcks; geom++) {
      if ((geom == PickResizeU) || (geom == PickResizeD))
	 picks[geom]->set_principal(Down);
      else if ((geom == PickResizeL) || (geom == PickResizeR))
	 picks[geom]->set_principal(Right);
      else if ((geom == PickSphL) || (geom == PickSphR))
	 picks[geom]->set_principal(Down, Norm);
      else if ((geom == PickSphU) || (geom == PickSphD))
	 picks[geom]->set_principal(Right, Norm);
      else
	 picks[geom]->set_principal(Right, Down, Norm);
   }
}

void
FrameWidget::geom_moved( GeomPick*, int axis, double dist,
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
	 delt = variables[CenterVar]->point() + GetRightAxis()*ResizeMin - variables[PointRVar]->point();
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
	 delt = variables[CenterVar]->point() + GetRightAxis()*ResizeMin - variables[PointRVar]->point();
      variables[PointDVar]->MoveDelta(delt/2.0);      
      variables[CenterVar]->SetDelta(delt/2.0, Scheme3);
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
   variables[CenterVar]->MoveDelta(delta);
   variables[PointRVar]->MoveDelta(delta);
   variables[PointDVar]->MoveDelta(delta);

   execute();
}


Point
FrameWidget::ReferencePoint() const
{
   return variables[CenterVar]->point();
}


void
FrameWidget::SetPosition( const Point& center, const Point& R, const Point& D )
{
   variables[PointRVar]->Move(R);
   variables[PointDVar]->Move(D);
   variables[DistRVar]->Move((R-center).length());
   variables[DistDVar]->Move((D-center).length());
   variables[CenterVar]->Set(center, Scheme3); // This should set Hypo...

   execute();
}


void
FrameWidget::GetPosition( Point& center, Point& R, Point& D )
{
   center = variables[CenterVar]->point();
   R = variables[PointRVar]->point();
   D = variables[PointDVar]->point();
}


void
FrameWidget::SetPosition( const Point& center, const Vector& normal,
			  const Real size1, const Real size2 )
{
   Vector axis1, axis2;
   normal.find_orthogonal(axis1, axis2);
   
   variables[PointRVar]->Move(center+axis1*size1);
   variables[PointDVar]->Move(center+axis2*size2);
   variables[CenterVar]->Move(center);
   variables[DistRVar]->Move(size1);
   variables[DistDVar]->Set(size2); // This should set the Hypo...

   execute();
}


void
FrameWidget::GetPosition( Point& center, Vector& normal,
			  Real& size1, Real& size2 )
{
   center = variables[CenterVar]->point();
   normal = Cross(GetRightAxis(), GetDownAxis());
   size1 = variables[DistRVar]->real();
   size2 = variables[DistDVar]->real();
}


void
FrameWidget::SetSize( const Real sizeR, const Real sizeD )
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

   execute();
}

void
FrameWidget::GetSize( Real& sizeR, Real& sizeD ) const
{
   sizeR = variables[DistRVar]->real();
   sizeD = variables[DistDVar]->real();
}


const Vector&
FrameWidget::GetRightAxis()
{
   Vector axis(variables[PointRVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldrightaxis;
   else
      return (oldrightaxis = axis.normal());
}


const Vector&
FrameWidget::GetDownAxis()
{
   Vector axis(variables[PointDVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return olddownaxis;
   else
      return (olddownaxis = axis.normal());
}


clString
FrameWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   
   switch(mindex){
   case 0:
      return "Point";
   case 1:
      return "Edge";
   case 2:
      return "Resize";
   default:
      return "UnknownMaterial";
   }
}


