
/*
 *  FFrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/FixedFrameWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Constraints/PlaneConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 9;
const Index NumVars = 8;
const Index NumGeoms = 10;
const Index NumMatls = 3;
const Index NumPcks = 7;
const Index NumSchemes = 5;

enum { FFrameW_ConstULDR, FFrameW_ConstURDL, FFrameW_ConstPyth, FFrameW_ConstPlane,
       FFrameW_ConstULUR, FFrameW_ConstULDL, FFrameW_ConstDRUR, FFrameW_ConstDRDL,
       FFrameW_ConstRatio };
enum { FFrameW_GeomPointUL, FFrameW_GeomPointUR, FFrameW_GeomPointDR, FFrameW_GeomPointDL,
       FFrameW_GeomCylU, FFrameW_GeomCylR, FFrameW_GeomCylD, FFrameW_GeomCylL,
       FFrameW_GeomResizeU, FFrameW_GeomResizeD };
enum { FFrameW_PickSphUL, FFrameW_PickSphUR, FFrameW_PickSphDR, FFrameW_PickSphDL, FFrameW_PickCyls,
       FFrameW_PickResizeU, FFrameW_PickResizeD };

FixedFrameWidget::FixedFrameWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1),
  oldaxis1(1, 0, 0), oldaxis2(0, 1, 0)
{
   Real INIT = 1.0*widget_scale;
   // Scheme5 is used by the resize picks in GeomMoved!!
   variables[FFrameW_PointUL] = new PointVariable("PntUL", Scheme1, Point(0, 0, 0));
   variables[FFrameW_PointUR] = new PointVariable("PntUR", Scheme2, Point(INIT, 0, 0));
   variables[FFrameW_PointDR] = new PointVariable("PntDR", Scheme3, Point(INIT, INIT, 0));
   variables[FFrameW_PointDL] = new PointVariable("PntDL", Scheme4, Point(0, INIT, 0));
   variables[FFrameW_Dist1] = new RealVariable("DIST1", Scheme1, INIT);
   variables[FFrameW_Dist2] = new RealVariable("DIST2", Scheme1, INIT);
   variables[FFrameW_Ratio] = new RealVariable("Ratio", Scheme1, 1.0);
   variables[FFrameW_Hypo] = new RealVariable("HYPO", Scheme1, sqrt(2*INIT*INIT));

   constraints[FFrameW_ConstRatio] = new RatioConstraint("ConstRatio",
							 NumSchemes,
							 variables[FFrameW_Dist1],
							 variables[FFrameW_Dist2],
							 variables[FFrameW_Ratio]);
   constraints[FFrameW_ConstRatio]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FFrameW_ConstRatio]->VarChoices(Scheme2, 1, 0, 0);
   constraints[FFrameW_ConstRatio]->VarChoices(Scheme3, 1, 0, 1);
   constraints[FFrameW_ConstRatio]->VarChoices(Scheme4, 1, 0, 0);
   constraints[FFrameW_ConstRatio]->VarChoices(Scheme5, 1, 0, 0);
   constraints[FFrameW_ConstRatio]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[FFrameW_ConstPlane] = new PlaneConstraint("ConstPlane",
							 NumSchemes,
							 variables[FFrameW_PointUL],
							 variables[FFrameW_PointUR],
							 variables[FFrameW_PointDR],
							 variables[FFrameW_PointDL]);
   constraints[FFrameW_ConstPlane]->VarChoices(Scheme1, 2, 3, 2, 1);
   constraints[FFrameW_ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 3);
   constraints[FFrameW_ConstPlane]->VarChoices(Scheme3, 0, 3, 0, 1);
   constraints[FFrameW_ConstPlane]->VarChoices(Scheme4, 2, 1, 0, 1);
   constraints[FFrameW_ConstPlane]->VarChoices(Scheme5, 2, 3, 0, 1);
   constraints[FFrameW_ConstPlane]->Priorities(P_Highest, P_Highest,
					       P_Highest, P_Highest);
   constraints[FFrameW_ConstULDR] = new DistanceConstraint("Const13",
							   NumSchemes,
							   variables[FFrameW_PointUL],
							   variables[FFrameW_PointDR],
							   variables[FFrameW_Hypo]);
   constraints[FFrameW_ConstULDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FFrameW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[FFrameW_ConstULDR]->VarChoices(Scheme3, 0, 0, 0);
   constraints[FFrameW_ConstULDR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[FFrameW_ConstULDR]->VarChoices(Scheme5, 1, 0, 1);
   constraints[FFrameW_ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[FFrameW_ConstURDL] = new DistanceConstraint("Const24",
							   NumSchemes,
							   variables[FFrameW_PointUR],
							   variables[FFrameW_PointDL],
							   variables[FFrameW_Hypo]);
   constraints[FFrameW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FFrameW_ConstURDL]->VarChoices(Scheme2, 1, 1, 1);
   constraints[FFrameW_ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[FFrameW_ConstURDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FFrameW_ConstURDL]->VarChoices(Scheme5, 2, 2, 0);
   constraints[FFrameW_ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[FFrameW_ConstPyth] = new PythagorasConstraint("ConstPyth",
							     NumSchemes,
							     variables[FFrameW_Dist1],
							     variables[FFrameW_Dist2],
							     variables[FFrameW_Hypo]);
   constraints[FFrameW_ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FFrameW_ConstPyth]->VarChoices(Scheme2, 1, 0, 1);
   constraints[FFrameW_ConstPyth]->VarChoices(Scheme3, 1, 0, 1);
   constraints[FFrameW_ConstPyth]->VarChoices(Scheme4, 1, 0, 1);
   constraints[FFrameW_ConstPyth]->VarChoices(Scheme5, 1, 0, 1);
   constraints[FFrameW_ConstPyth]->Priorities(P_Default, P_Default, P_HighMedium);
   constraints[FFrameW_ConstULUR] = new DistanceConstraint("Const12",
							   NumSchemes,
							   variables[FFrameW_PointUL],
							   variables[FFrameW_PointUR],
							   variables[FFrameW_Dist1]);
   constraints[FFrameW_ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FFrameW_ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FFrameW_ConstULUR]->VarChoices(Scheme3, 1, 0, 0);
   constraints[FFrameW_ConstULUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[FFrameW_ConstULUR]->VarChoices(Scheme5, 0, 0, 0);
   constraints[FFrameW_ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FFrameW_ConstULDL] = new DistanceConstraint("Const14",
							   NumSchemes,
							   variables[FFrameW_PointUL],
							   variables[FFrameW_PointDL],
							   variables[FFrameW_Dist2]);
   constraints[FFrameW_ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FFrameW_ConstULDL]->VarChoices(Scheme2, 1, 0, 0);
   constraints[FFrameW_ConstULDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[FFrameW_ConstULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FFrameW_ConstULDL]->VarChoices(Scheme5, 0, 0, 0);
   constraints[FFrameW_ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FFrameW_ConstDRUR] = new DistanceConstraint("Const32",
							   NumSchemes,
							   variables[FFrameW_PointDR],
							   variables[FFrameW_PointUR],
							   variables[FFrameW_Dist2]);
   constraints[FFrameW_ConstDRUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FFrameW_ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FFrameW_ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[FFrameW_ConstDRUR]->VarChoices(Scheme4, 1, 0, 0);
   constraints[FFrameW_ConstDRUR]->VarChoices(Scheme5, 0, 0, 0);
   constraints[FFrameW_ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FFrameW_ConstDRDL] = new DistanceConstraint("Const34",
							   NumSchemes,
							   variables[FFrameW_PointDR],
							   variables[FFrameW_PointDL],
							   variables[FFrameW_Dist1]);
   constraints[FFrameW_ConstDRDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FFrameW_ConstDRDL]->VarChoices(Scheme2, 1, 0, 0);
   constraints[FFrameW_ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[FFrameW_ConstDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FFrameW_ConstDRDL]->VarChoices(Scheme5, 0, 0, 0);
   constraints[FFrameW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[FFrameW_PointMatl] = PointWidgetMaterial;
   materials[FFrameW_EdgeMatl] = EdgeWidgetMaterial;
   materials[FFrameW_HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = FFrameW_GeomPointUL, pick = FFrameW_PickSphUL;
	geom <= FFrameW_GeomPointDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[FFrameW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[FFrameW_PointMatl]);
   
   GeomGroup* resizes = new GeomGroup;
   geometries[FFrameW_GeomResizeU] = new GeomCappedCylinder;
   picks[FFrameW_PickResizeU] = new GeomPick(geometries[FFrameW_GeomResizeU], module);
   picks[FFrameW_PickResizeU]->set_highlight(materials[FFrameW_HighMatl]);
   picks[FFrameW_PickResizeU]->set_cbdata((void*)FFrameW_PickResizeU);
   resizes->add(picks[FFrameW_PickResizeU]);
   geometries[FFrameW_GeomResizeD] = new GeomCappedCylinder;
   picks[FFrameW_PickResizeD] = new GeomPick(geometries[FFrameW_GeomResizeD], module);
   picks[FFrameW_PickResizeD]->set_highlight(materials[FFrameW_HighMatl]);
   picks[FFrameW_PickResizeD]->set_cbdata((void*)FFrameW_PickResizeD);
   resizes->add(picks[FFrameW_PickResizeD]);
   GeomMaterial* resizem = new GeomMaterial(resizes, materials[FFrameW_PointMatl]);
   
   GeomGroup* cyls = new GeomGroup;
   for (geom = FFrameW_GeomCylU; geom <= FFrameW_GeomCylL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[FFrameW_PickCyls] = new GeomPick(cyls, module);
   picks[FFrameW_PickCyls]->set_highlight(materials[FFrameW_HighMatl]);
   picks[FFrameW_PickCyls]->set_cbdata((void*)FFrameW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[FFrameW_PickCyls], materials[FFrameW_EdgeMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(resizem);
   w->add(cylsm);

   SetEpsilon(widget_scale*1e-6);
   
   FinishWidget(w);
}


FixedFrameWidget::~FixedFrameWidget()
{
}


void
FixedFrameWidget::widget_execute()
{
   ((GeomSphere*)geometries[FFrameW_GeomPointUL])->move(variables[FFrameW_PointUL]->GetPoint(),
							1*widget_scale);
   ((GeomSphere*)geometries[FFrameW_GeomPointUR])->move(variables[FFrameW_PointUR]->GetPoint(),
							1*widget_scale);
   ((GeomSphere*)geometries[FFrameW_GeomPointDR])->move(variables[FFrameW_PointDR]->GetPoint(),
							1*widget_scale);
   ((GeomSphere*)geometries[FFrameW_GeomPointDL])->move(variables[FFrameW_PointDL]->GetPoint(),
							1*widget_scale);
   Point p(variables[FFrameW_PointUL]->GetPoint() + (variables[FFrameW_PointUR]->GetPoint()
						- variables[FFrameW_PointUL]->GetPoint()) / 2.0);
   ((GeomCappedCylinder*)geometries[FFrameW_GeomResizeU])->move(p - (GetAxis2() * 0.6 * widget_scale),
								p + (GetAxis2() * 0.6 * widget_scale),
								0.75*widget_scale);
   p = variables[FFrameW_PointDR]->GetPoint() + (variables[FFrameW_PointDL]->GetPoint()
					    - variables[FFrameW_PointDR]->GetPoint()) / 2.0;
   ((GeomCappedCylinder*)geometries[FFrameW_GeomResizeD])->move(p - (GetAxis2() * 0.6 * widget_scale),
								p + (GetAxis2() * 0.6 * widget_scale),
								0.75*widget_scale);
   ((GeomCylinder*)geometries[FFrameW_GeomCylU])->move(variables[FFrameW_PointUL]->GetPoint(),
						       variables[FFrameW_PointUR]->GetPoint(),
						       0.5*widget_scale);
   ((GeomCylinder*)geometries[FFrameW_GeomCylR])->move(variables[FFrameW_PointUR]->GetPoint(),
						       variables[FFrameW_PointDR]->GetPoint(),
						       0.5*widget_scale);
   ((GeomCylinder*)geometries[FFrameW_GeomCylD])->move(variables[FFrameW_PointDR]->GetPoint(),
						       variables[FFrameW_PointDL]->GetPoint(),
						       0.5*widget_scale);
   ((GeomCylinder*)geometries[FFrameW_GeomCylL])->move(variables[FFrameW_PointDL]->GetPoint(),
						       variables[FFrameW_PointUL]->GetPoint(),
						       0.5*widget_scale);

   ((DistanceConstraint*)constraints[FFrameW_ConstULUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[FFrameW_ConstDRDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[FFrameW_ConstULDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[FFrameW_ConstDRUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[FFrameW_ConstULDR])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);
   ((DistanceConstraint*)constraints[FFrameW_ConstURDL])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);

   SetEpsilon(widget_scale*1e-6);

   Vector spvec1(variables[FFrameW_PointUR]->GetPoint() - variables[FFrameW_PointUL]->GetPoint());
   Vector spvec2(variables[FFrameW_PointDL]->GetPoint() - variables[FFrameW_PointUL]->GetPoint());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if ((geom == FFrameW_PickResizeU) || (geom == FFrameW_PickResizeD))
	    picks[geom]->set_principal(spvec2);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
FixedFrameWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			      void* cbdata )
{
   Vector delt(delta);
   Real t;
   
   ((DistanceConstraint*)constraints[FFrameW_ConstULUR])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[FFrameW_ConstDRDL])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[FFrameW_ConstULDL])->SetDefault(GetAxis2());
   ((DistanceConstraint*)constraints[FFrameW_ConstDRUR])->SetDefault(GetAxis2());

   for (Index v=0; v<NumVars; v++)
      variables[v]->Reset();
   
   switch((int)cbdata){
   case FFrameW_PickSphUL:
      variables[FFrameW_PointUL]->SetDelta(delta);
      break;
   case FFrameW_PickSphUR:
      variables[FFrameW_PointUR]->SetDelta(delta);
      break;
   case FFrameW_PickSphDR:
      variables[FFrameW_PointDR]->SetDelta(delta);
      break;
   case FFrameW_PickSphDL:
      variables[FFrameW_PointDL]->SetDelta(delta);
      break;
   case FFrameW_PickResizeU:
      if (((variables[FFrameW_PointUL]->GetPoint()+delta)-variables[FFrameW_PointDL]->GetPoint()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[FFrameW_PointDL]->GetPoint() + delta.normal()*3.2*widget_scale)
		 - variables[FFrameW_PointUL]->GetPoint());
      }
      t = delt.length();
      if (Dot(delt, GetAxis2()) < 0.0)
	 t = -t;
      variables[FFrameW_PointDL]->MoveDelta(GetAxis1()*t/2.0);
      variables[FFrameW_PointUR]->SetDelta(delt-GetAxis1()*t/2.0, Scheme5);
      break;
   case FFrameW_PickResizeD:
      if (((variables[FFrameW_PointDR]->GetPoint()+delta)-variables[FFrameW_PointUR]->GetPoint()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[FFrameW_PointUR]->GetPoint() + delta.normal()*3.2*widget_scale)
		 - variables[FFrameW_PointDR]->GetPoint());
      }
      t = delt.length();
      if (Dot(delt, GetAxis2()) < 0.0)
	 t = -t;
      variables[FFrameW_PointUR]->MoveDelta(GetAxis1()*t/2.0);
      variables[FFrameW_PointDL]->SetDelta(delt-GetAxis1()*t/2.0, Scheme5);
      break;
   case FFrameW_PickCyls:
      variables[FFrameW_PointUL]->MoveDelta(delta);
      variables[FFrameW_PointUR]->MoveDelta(delta);
      variables[FFrameW_PointDR]->MoveDelta(delta);
      variables[FFrameW_PointDL]->MoveDelta(delta);
      break;
   }
}

