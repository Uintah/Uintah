
/*
 *  BoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/BoxWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 9;
const Index NumVars = 10;
const Index NumGeoms = 32;
const Index NumPcks = 13;
const Index NumMatls = 3;
const Index NumMdes = 2;
const Index NumSwtchs = 3;
const Index NumSchemes = 6;

enum { ConstRD, ConstDI, ConstIR, ConstRC, ConstDC, ConstIC,
       ConstPythRD, ConstPythDI, ConstPythIR };
enum { SphereR, SphereL, SphereD, SphereU, SphereI, SphereO,
       SmallSphereIUL, SmallSphereIUR, SmallSphereIDR, SmallSphereIDL,
       SmallSphereOUL, SmallSphereOUR, SmallSphereODR, SmallSphereODL,
       CylIU, CylIR, CylID, CylIL,
       CylMU, CylMR, CylMD, CylML,
       CylOU, CylOR, CylOD, CylOL,
       GeomResizeR, GeomResizeL, GeomResizeD, GeomResizeU,
       GeomResizeI, GeomResizeO };
enum { PickSphR, PickSphL, PickSphD, PickSphU, PickSphI, PickSphO,
       PickCyls,
       PickResizeR, PickResizeL, PickResizeD, PickResizeU,
       PickResizeI, PickResizeO };

BoxWidget::BoxWidget( Module* module, CrowdMonitor* lock, double widget_scale,
		      Index aligned )
: BaseWidget(module, lock, "BoxWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  oldrightaxis(1, 0, 0), olddownaxis(0, 1, 0), oldinaxis(0, 0, 1),
  aligned(aligned)
{
   Real INIT = 5.0*widget_scale;
   variables[CenterVar] = new PointVariable("Center", solve, Scheme1, Point(0, 0, 0));
   variables[PointRVar] = new PointVariable("PntR", solve, Scheme1, Point(INIT, 0, 0));
   variables[PointDVar] = new PointVariable("PntD", solve, Scheme2, Point(0, INIT, 0));
   variables[PointIVar] = new PointVariable("PntI", solve, Scheme3, Point(0, 0, INIT));
   variables[DistRVar] = new RealVariable("DISTR", solve, Scheme4, INIT);
   variables[DistDVar] = new RealVariable("DISTD", solve, Scheme5, INIT);
   variables[DistIVar] = new RealVariable("DISTI", solve, Scheme6, INIT);
   variables[HypoRDVar] = new RealVariable("HYPOR", solve, Scheme4, sqrt(2*INIT*INIT));
   variables[HypoDIVar] = new RealVariable("HYPOD", solve, Scheme5, sqrt(2*INIT*INIT));
   variables[HypoIRVar] = new RealVariable("HYPOI", solve, Scheme6, sqrt(2*INIT*INIT));

   constraints[ConstRD] = new DistanceConstraint("ConstRD",
						 NumSchemes,
						 variables[PointRVar],
						 variables[PointDVar],
						 variables[HypoRDVar]);
   constraints[ConstRD]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ConstRD]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRD]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRD]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstRD]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstRD]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstRD]->Priorities(P_Default, P_Default, P_Default);
   constraints[ConstPythRD] = new PythagorasConstraint("ConstPythRD",
						     NumSchemes,
						     variables[DistRVar],
						     variables[DistDVar],
						     variables[HypoRDVar]);
   constraints[ConstPythRD]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPythRD]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstPythRD]->VarChoices(Scheme3, 1, 0, 0);
   constraints[ConstPythRD]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstPythRD]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstPythRD]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstPythRD]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstDI] = new DistanceConstraint("ConstDI",
						 NumSchemes,
						 variables[PointDVar],
						 variables[PointIVar],
						 variables[HypoDIVar]);
   constraints[ConstDI]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstDI]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ConstDI]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstDI]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstDI]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstDI]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstDI]->Priorities(P_Default, P_Default, P_Default);
   constraints[ConstPythDI] = new PythagorasConstraint("ConstPythDI",
						     NumSchemes,
						     variables[DistDVar],
						     variables[DistIVar],
						     variables[HypoDIVar]);
   constraints[ConstPythDI]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPythDI]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstPythDI]->VarChoices(Scheme3, 1, 0, 0);
   constraints[ConstPythDI]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstPythDI]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstPythDI]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstPythDI]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstIR] = new DistanceConstraint("ConstIR",
						 NumSchemes,
						 variables[PointIVar],
						 variables[PointRVar],
						 variables[HypoIRVar]);
   constraints[ConstIR]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstIR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ConstIR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstIR]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstIR]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstIR]->Priorities(P_Default, P_Default, P_Default);
   constraints[ConstPythIR] = new PythagorasConstraint("ConstPythIR",
						     NumSchemes,
						     variables[DistIVar],
						     variables[DistRVar],
						     variables[HypoIRVar]);
   constraints[ConstPythIR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ConstPythIR]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ConstPythIR]->VarChoices(Scheme3, 1, 0, 0);
   constraints[ConstPythIR]->VarChoices(Scheme4, 2, 2, 1);
   constraints[ConstPythIR]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ConstPythIR]->VarChoices(Scheme6, 2, 2, 0);
   constraints[ConstPythIR]->Priorities(P_Highest, P_Highest, P_Highest);
   constraints[ConstRC] = new DistanceConstraint("ConstRC",
						 NumSchemes,
						 variables[PointRVar],
						 variables[CenterVar],
						 variables[DistRVar]);
   constraints[ConstRC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme4, 2, 2, 2);
   constraints[ConstRC]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ConstRC]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstRC]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstDC] = new DistanceConstraint("ConstDC",
					       NumSchemes,
					       variables[PointDVar],
					       variables[CenterVar],
					       variables[DistDVar]);
   constraints[ConstDC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstDC]->VarChoices(Scheme5, 2, 2, 2);
   constraints[ConstDC]->VarChoices(Scheme6, 0, 0, 0);
   constraints[ConstDC]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ConstIC] = new DistanceConstraint("ConstIC",
					       NumSchemes,
					       variables[PointIVar],
					       variables[CenterVar],
					       variables[DistIVar]);
   constraints[ConstIC]->VarChoices(Scheme1, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ConstIC]->VarChoices(Scheme6, 2, 2, 2);
   constraints[ConstIC]->Priorities(P_Highest, P_Highest, P_Default);

   Index geom, pick;
   GeomGroup* cyls = new GeomGroup;
   for (geom = SmallSphereIUL; geom <= SmallSphereODL; geom++) {
      geometries[geom] = new GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = CylIU; geom <= CylOL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[PickCyls] = new GeomPick(cyls, module, this, PickCyls);
   picks[PickCyls]->set_highlight(DefaultHighlightMaterial);
   materials[EdgeMatl] = new GeomMaterial(picks[PickCyls], DefaultEdgeMaterial);
   CreateModeSwitch(0, materials[EdgeMatl]);

   GeomGroup* pts = new GeomGroup;
   for (geom = SphereR, pick = PickSphR;
	geom <= SphereO; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module, this, pick);
      picks[pick]->set_highlight(DefaultHighlightMaterial);
      pts->add(picks[pick]);
   }
   materials[PointMatl] = new GeomMaterial(pts, DefaultPointMaterial);
   CreateModeSwitch(1, materials[PointMatl]);
   
   GeomGroup* resizes = new GeomGroup;
   for (geom = GeomResizeR, pick = PickResizeR;
	geom <= GeomResizeO; geom++, pick++) {
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


BoxWidget::~BoxWidget()
{
}


void
BoxWidget::widget_execute()
{
   Real sphererad(widget_scale), resizerad(0.5*widget_scale), cylinderrad(0.5*widget_scale);
   Vector Right(GetRightAxis()*variables[DistRVar]->real());
   Vector Down(GetDownAxis()*variables[DistDVar]->real());
   Vector In(GetInAxis()*variables[DistIVar]->real());
   Point Center(variables[CenterVar]->point());
   Point IUL(Center-Right-Down+In);
   Point IUR(Center+Right-Down+In);
   Point IDR(Center+Right+Down+In);
   Point IDL(Center-Right+Down+In);
   Point OUL(Center-Right-Down-In);
   Point OUR(Center+Right-Down-In);
   Point ODR(Center+Right+Down-In);
   Point ODL(Center-Right+Down-In);
   Point U(Center-Down);
   Point R(Center+Right);
   Point D(Center+Down);
   Point L(Center-Right);
   Point I(Center+In);
   Point O(Center-In);
   
   if (mode_switches[0]->get_state()) {
      ((GeomCylinder*)geometries[CylIU])->move(IUL, IUR, cylinderrad);
      ((GeomCylinder*)geometries[CylIR])->move(IUR, IDR, cylinderrad);
      ((GeomCylinder*)geometries[CylID])->move(IDR, IDL, cylinderrad);
      ((GeomCylinder*)geometries[CylIL])->move(IDL, IUL, cylinderrad);
      ((GeomCylinder*)geometries[CylMU])->move(IUL, OUL, cylinderrad);
      ((GeomCylinder*)geometries[CylMR])->move(IUR, OUR, cylinderrad);
      ((GeomCylinder*)geometries[CylMD])->move(IDR, ODR, cylinderrad);
      ((GeomCylinder*)geometries[CylML])->move(IDL, ODL, cylinderrad);
      ((GeomCylinder*)geometries[CylOU])->move(OUL, OUR, cylinderrad);
      ((GeomCylinder*)geometries[CylOR])->move(OUR, ODR, cylinderrad);
      ((GeomCylinder*)geometries[CylOD])->move(ODR, ODL, cylinderrad);
      ((GeomCylinder*)geometries[CylOL])->move(ODL, OUL, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereIUL])->move(IUL, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereIUR])->move(IUR, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereIDR])->move(IDR, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereIDL])->move(IDL, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereOUL])->move(OUL, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereOUR])->move(OUR, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereODR])->move(ODR, cylinderrad);
      ((GeomSphere*)geometries[SmallSphereODL])->move(ODL, cylinderrad);
   }

   if ((aligned == 0) && (mode_switches[1]->get_state())) {
      ((GeomSphere*)geometries[SphereR])->move(R, sphererad);
      ((GeomSphere*)geometries[SphereL])->move(L, sphererad);
      ((GeomSphere*)geometries[SphereD])->move(D, sphererad);
      ((GeomSphere*)geometries[SphereU])->move(U, sphererad);
      ((GeomSphere*)geometries[SphereI])->move(I, sphererad);
      ((GeomSphere*)geometries[SphereO])->move(O, sphererad);
   }

   if (mode_switches[2]->get_state()) {
      Vector resizeR(GetRightAxis()*1.5*widget_scale),
	 resizeD(GetDownAxis()*1.5*widget_scale),
	 resizeI(GetInAxis()*1.5*widget_scale);
      
      ((GeomCappedCylinder*)geometries[GeomResizeR])->move(R, R + resizeR, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeL])->move(L, L - resizeR, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeD])->move(D, D + resizeD, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeU])->move(U, U - resizeD, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeI])->move(I, I + resizeI, resizerad);
      ((GeomCappedCylinder*)geometries[GeomResizeO])->move(O, O - resizeI, resizerad);
   }

   Right.normalize();
   Down.normalize();
   In.normalize();
   for (Index geom = 0; geom < NumPcks; geom++) {
      if ((geom == PickResizeU) || (geom == PickResizeD))
	 picks[geom]->set_principal(Down);
      else if ((geom == PickResizeL) || (geom == PickResizeR))
	 picks[geom]->set_principal(Right);
      else if ((geom == PickResizeO) || (geom == PickResizeI))
	 picks[geom]->set_principal(In);
      else if ((geom == PickSphL) || (geom == PickSphR))
	 picks[geom]->set_principal(Down, In);
      else if ((geom == PickSphU) || (geom == PickSphD))
	 picks[geom]->set_principal(Right, In);
      else if ((geom == PickSphO) || (geom == PickSphI))
	 picks[geom]->set_principal(Right, Down);
      else
	 picks[geom]->set_principal(Right, Down, In);
   }
}

void
BoxWidget::geom_moved( int /* axis*/, double /*dist*/, const Vector& delta,
		       int pick, const BState& )
{
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
   case PickSphI:
      variables[PointIVar]->SetDelta(delta);
      break;
   case PickSphO:
      variables[PointIVar]->SetDelta(-delta);
      break;
   case PickResizeU:
      variables[PointRVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->MoveDelta(delta/2.0);
      variables[CenterVar]->SetDelta(delta/2.0, Scheme5);
      break;
   case PickResizeR:
      variables[CenterVar]->MoveDelta(delta/2.0);
      variables[PointDVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->MoveDelta(delta/2.0);
      variables[PointRVar]->SetDelta(delta, Scheme4);
      break;
   case PickResizeD:
      variables[CenterVar]->MoveDelta(delta/2.0);
      variables[PointRVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->MoveDelta(delta/2.0);
      variables[PointDVar]->SetDelta(delta, Scheme5);
      break;
   case PickResizeL:
      variables[PointDVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->MoveDelta(delta/2.0);
      variables[CenterVar]->SetDelta(delta/2.0, Scheme4);
      break;
   case PickResizeI:
      variables[CenterVar]->MoveDelta(delta/2.0);
      variables[PointRVar]->MoveDelta(delta/2.0);
      variables[PointDVar]->MoveDelta(delta/2.0);
      variables[PointIVar]->SetDelta(delta, Scheme6);
      break;
   case PickResizeO:
      variables[PointRVar]->MoveDelta(delta/2.0);
      variables[PointDVar]->MoveDelta(delta/2.0);
      variables[CenterVar]->SetDelta(delta/2.0, Scheme6);
      break;
   case PickCyls:
      MoveDelta(delta);
      break;
   }
   execute();
}


void
BoxWidget::MoveDelta( const Vector& delta )
{
   variables[CenterVar]->MoveDelta(delta);
   variables[PointRVar]->MoveDelta(delta);
   variables[PointDVar]->MoveDelta(delta);
   variables[PointIVar]->MoveDelta(delta);

   execute();
}


Point
BoxWidget::ReferencePoint() const
{
   return variables[CenterVar]->point();
}


const Vector&
BoxWidget::GetRightAxis()
{
   Vector axis(variables[PointRVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldrightaxis;
   else
      return (oldrightaxis = axis.normal());
}


const Vector&
BoxWidget::GetDownAxis()
{
   Vector axis(variables[PointDVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return olddownaxis;
   else
      return (olddownaxis = axis.normal());
}


const Vector&
BoxWidget::GetInAxis()
{
   Vector axis(variables[PointIVar]->point() - variables[CenterVar]->point());
   if (axis.length2() <= 1e-6)
      return oldinaxis;
   else
      return (oldinaxis = axis.normal());
}


clString
BoxWidget::GetMaterialName( const Index mindex ) const
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


Index
BoxWidget::IsAxisAligned() const
{
   return aligned;
}


void
BoxWidget::AxisAligned( const Index yesno )
{
   if (aligned == yesno) return;
   
   aligned = yesno;

   if (aligned) {
      Point center(variables[CenterVar]->point());
      // Shouldn't need to resolve constraints...
      variables[PointRVar]->Move(center+Vector(1,0,0)*variables[DistRVar]->real());
      variables[PointDVar]->Move(center+Vector(0,1,0)*variables[DistDVar]->real());
      variables[PointIVar]->Move(center+Vector(0,0,1)*variables[DistIVar]->real());
      oldrightaxis = Vector(1,0,0);
      olddownaxis = Vector(0,1,0);
      oldinaxis = Vector(0,0,1);
   }
   
   execute();
}

