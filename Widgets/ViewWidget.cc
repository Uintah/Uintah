
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
#include <Constraints/PythagorasConstraint.h>
#include <Constraints/PlaneConstraint.h>
#include <Constraints/RatioConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 9;
const Index NumVars = 13;
const Index NumGeoms = 26;
const Index NumMatls = 4;
const Index NumPcks = 10;
const Index NumSchemes = 5;

enum { ViewW_ConstULDR, ViewW_ConstURDL, ViewW_ConstPyth, ViewW_ConstPlane,
       ViewW_ConstULUR, ViewW_ConstULDL, ViewW_ConstDRUR, ViewW_ConstDRDL,
       ViewW_ConstRatio };
enum { ViewW_GeomPointUL, ViewW_GeomPointUR, ViewW_GeomPointDR, ViewW_GeomPointDL,
       ViewW_GeomCylU, ViewW_GeomCylR, ViewW_GeomCylD, ViewW_GeomCylL,
       ViewW_GeomResizeU, ViewW_GeomResizeD,
       ViewW_GeomEye, ViewW_GeomForeEye, ViewW_GeomBackEye, ViewW_GeomShaft,
       ViewW_GeomCornerUL, ViewW_GeomCornerUR, ViewW_GeomCornerDR, ViewW_GeomCornerDL,
       ViewW_GeomEdgeU, ViewW_GeomEdgeR, ViewW_GeomEdgeD, ViewW_GeomEdgeL,
       ViewW_GeomDiagUL, ViewW_GeomDiagUR, ViewW_GeomDiagDR, ViewW_GeomDiagDL };
enum { ViewW_PickSphUL, ViewW_PickSphUR, ViewW_PickSphDR, ViewW_PickSphDL, ViewW_PickCyls,
       ViewW_PickResizeU, ViewW_PickResizeD, ViewW_PickEye, ViewW_PickForeEye, ViewW_PickBackEye };

ViewWidget::ViewWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1),
  oldaxis1(1, 0, 0), oldaxis2(1, 0, 0)
{
   Real INIT = 1.0*widget_scale;
   // Scheme2/3 are used by the picks in GeomMoved!!
   variables[ViewW_PointUL] = new Variable("PntUL", Scheme1, Point(0, 0, 0));
   variables[ViewW_PointUR] = new Variable("PntUR", Scheme2, Point(INIT, 0, 0));
   variables[ViewW_PointDR] = new Variable("PntDR", Scheme3, Point(INIT, INIT, 0));
   variables[ViewW_PointDL] = new Variable("PntDL", Scheme4, Point(0, INIT, 0));
   variables[ViewW_Eye] = new Variable("Eye", Scheme4, Point(INIT/2, INIT/2, -2*INIT));
   variables[ViewW_ForeEye] = new Variable("ForeEye", Scheme4, Point(INIT/2, INIT/2, 0));
   variables[ViewW_BackEye] = new Variable("BackEye", Scheme4, Point(INIT/2, INIT/2, 3*INIT));
   variables[ViewW_Dist1] = new Variable("DIST1", Scheme1, Point(INIT, 0, 0));
   variables[ViewW_Dist2] = new Variable("DIST2", Scheme1, Point(INIT, 0, 0));
   variables[ViewW_Ratio] = new Variable("Ratio", Scheme1, Point(1.0, 0, 0));
   variables[ViewW_Hypo] = new Variable("HYPO", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));
   variables[ViewW_Fore] = new Variable("Fore", Scheme1, Point(2.0, 0, 0));
   variables[ViewW_Back] = new Variable("Back", Scheme1, Point(5.0, 0, 0));

   constraints[ViewW_ConstRatio] = new RatioConstraint("ConstRatio",
						       NumSchemes,
						       variables[ViewW_Dist1],
						       variables[ViewW_Dist2],
						       variables[ViewW_Ratio]);
   constraints[ViewW_ConstRatio]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ViewW_ConstRatio]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ViewW_ConstRatio]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ViewW_ConstRatio]->VarChoices(Scheme4, 1, 0, 0);
   constraints[ViewW_ConstRatio]->VarChoices(Scheme5, 1, 0, 0);
   constraints[ViewW_ConstRatio]->Priorities(P_Highest, P_Highest, P_Default);
   constraints[ViewW_ConstPlane] = new PlaneConstraint("ConstPlane",
						       NumSchemes,
						       variables[ViewW_PointUL],
						       variables[ViewW_PointUR],
						       variables[ViewW_PointDR],
						       variables[ViewW_PointDL]);
   constraints[ViewW_ConstPlane]->VarChoices(Scheme1, 2, 3, 2, 1);
   constraints[ViewW_ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 3);
   constraints[ViewW_ConstPlane]->VarChoices(Scheme3, 0, 3, 0, 1);
   constraints[ViewW_ConstPlane]->VarChoices(Scheme4, 2, 1, 0, 1);
   constraints[ViewW_ConstPlane]->VarChoices(Scheme5, 2, 3, 0, 1);
   constraints[ViewW_ConstPlane]->Priorities(P_Highest, P_Highest,
					     P_Highest, P_Highest);
   constraints[ViewW_ConstULDR] = new DistanceConstraint("Const13",
							 NumSchemes,
							 variables[ViewW_PointUL],
							 variables[ViewW_PointDR],
							 variables[ViewW_Hypo]);
   constraints[ViewW_ConstULDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ViewW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ViewW_ConstULDR]->VarChoices(Scheme3, 0, 0, 0);
   constraints[ViewW_ConstULDR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ViewW_ConstULDR]->VarChoices(Scheme5, 1, 0, 1);
   constraints[ViewW_ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[ViewW_ConstURDL] = new DistanceConstraint("Const24",
							 NumSchemes,
							 variables[ViewW_PointUR],
							 variables[ViewW_PointDL],
							 variables[ViewW_Hypo]);
   constraints[ViewW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ViewW_ConstURDL]->VarChoices(Scheme2, 1, 1, 1);
   constraints[ViewW_ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ViewW_ConstURDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ViewW_ConstURDL]->VarChoices(Scheme5, 2, 2, 0);
   constraints[ViewW_ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[ViewW_ConstPyth] = new PythagorasConstraint("ConstPyth",
							   NumSchemes,
							   variables[ViewW_Dist1],
							   variables[ViewW_Dist2],
							   variables[ViewW_Hypo]);
   constraints[ViewW_ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ViewW_ConstPyth]->VarChoices(Scheme2, 1, 0, 1);
   constraints[ViewW_ConstPyth]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ViewW_ConstPyth]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ViewW_ConstPyth]->VarChoices(Scheme5, 2, 2, 1);
   constraints[ViewW_ConstPyth]->Priorities(P_Default, P_Default, P_HighMedium);
   constraints[ViewW_ConstULUR] = new DistanceConstraint("Const12",
							 NumSchemes,
							 variables[ViewW_PointUL],
							 variables[ViewW_PointUR],
							 variables[ViewW_Dist1]);
   constraints[ViewW_ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ViewW_ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ViewW_ConstULUR]->VarChoices(Scheme3, 1, 0, 0);
   constraints[ViewW_ConstULUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[ViewW_ConstULUR]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ViewW_ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ViewW_ConstULDL] = new DistanceConstraint("Const14",
							 NumSchemes,
							 variables[ViewW_PointUL],
							 variables[ViewW_PointDL],
							 variables[ViewW_Dist2]);
   constraints[ViewW_ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[ViewW_ConstULDL]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ViewW_ConstULDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[ViewW_ConstULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ViewW_ConstULDL]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ViewW_ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ViewW_ConstDRUR] = new DistanceConstraint("Const32",
							 NumSchemes,
							 variables[ViewW_PointDR],
							 variables[ViewW_PointUR],
							 variables[ViewW_Dist2]);
   constraints[ViewW_ConstDRUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ViewW_ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[ViewW_ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ViewW_ConstDRUR]->VarChoices(Scheme4, 1, 0, 0);
   constraints[ViewW_ConstDRUR]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ViewW_ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[ViewW_ConstDRDL] = new DistanceConstraint("Const34",
							 NumSchemes,
							 variables[ViewW_PointDR],
							 variables[ViewW_PointDL],
							 variables[ViewW_Dist1]);
   constraints[ViewW_ConstDRDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[ViewW_ConstDRDL]->VarChoices(Scheme2, 1, 0, 0);
   constraints[ViewW_ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[ViewW_ConstDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[ViewW_ConstDRDL]->VarChoices(Scheme5, 0, 0, 0);
   constraints[ViewW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[ViewW_PointMatl] = PointWidgetMaterial;
   materials[ViewW_EdgeMatl] = EdgeWidgetMaterial;
   materials[ViewW_SpecialMatl] = SpecialWidgetMaterial;
   materials[ViewW_HighMatl] = HighlightWidgetMaterial;

   GeomGroup* eyes = new GeomGroup;
   geometries[ViewW_GeomEye] = new GeomSphere;
   picks[ViewW_PickEye] = new GeomPick(geometries[ViewW_GeomEye], module);
   picks[ViewW_PickEye]->set_highlight(materials[ViewW_HighMatl]);
   picks[ViewW_PickEye]->set_cbdata((void*)ViewW_PickEye);
   eyes->add(picks[ViewW_PickEye]);

   geometries[ViewW_GeomForeEye] = new GeomSphere;
   picks[ViewW_PickForeEye] = new GeomPick(geometries[ViewW_GeomForeEye], module);
   picks[ViewW_PickForeEye]->set_highlight(materials[ViewW_HighMatl]);
   picks[ViewW_PickForeEye]->set_cbdata((void*)ViewW_PickForeEye);
   eyes->add(picks[ViewW_PickForeEye]);

   geometries[ViewW_GeomBackEye] = new GeomSphere;
   picks[ViewW_PickBackEye] = new GeomPick(geometries[ViewW_GeomBackEye], module);
   picks[ViewW_PickBackEye]->set_highlight(materials[ViewW_HighMatl]);
   picks[ViewW_PickBackEye]->set_cbdata((void*)ViewW_PickBackEye);
   eyes->add(picks[ViewW_PickBackEye]);
   GeomMaterial* eyesm = new GeomMaterial(eyes, materials[ViewW_SpecialMatl]);

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = ViewW_GeomPointUL, pick = ViewW_PickSphUL;
	geom <= ViewW_GeomPointDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[ViewW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[ViewW_PointMatl]);
   
   GeomGroup* resizes = new GeomGroup;
   geometries[ViewW_GeomResizeU] = new GeomCappedCylinder;
   picks[ViewW_PickResizeU] = new GeomPick(geometries[ViewW_GeomResizeU], module);
   picks[ViewW_PickResizeU]->set_highlight(materials[ViewW_HighMatl]);
   picks[ViewW_PickResizeU]->set_cbdata((void*)ViewW_PickResizeU);
   resizes->add(picks[ViewW_PickResizeU]);
   geometries[ViewW_GeomResizeD] = new GeomCappedCylinder;
   picks[ViewW_PickResizeD] = new GeomPick(geometries[ViewW_GeomResizeD], module);
   picks[ViewW_PickResizeD]->set_highlight(materials[ViewW_HighMatl]);
   picks[ViewW_PickResizeD]->set_cbdata((void*)ViewW_PickResizeD);
   resizes->add(picks[ViewW_PickResizeD]);
   GeomMaterial* resizem = new GeomMaterial(resizes, materials[ViewW_PointMatl]);
   
   GeomGroup* cyls = new GeomGroup;
   geometries[ViewW_GeomShaft] = new GeomCylinder;
   cyls->add(geometries[ViewW_GeomShaft]);
   for (geom = ViewW_GeomCylU; geom <= ViewW_GeomCylL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   for (geom = ViewW_GeomCornerUL; geom <= ViewW_GeomCornerDL; geom++) {
      geometries[geom] = new GeomSphere;
      cyls->add(geometries[geom]);
   }
   for (geom = ViewW_GeomEdgeU; geom <= ViewW_GeomEdgeL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   for (geom = ViewW_GeomDiagUL; geom <= ViewW_GeomDiagDL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[ViewW_PickCyls] = new GeomPick(cyls, module);
   picks[ViewW_PickCyls]->set_highlight(materials[ViewW_HighMatl]);
   picks[ViewW_PickCyls]->set_cbdata((void*)ViewW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[ViewW_PickCyls], materials[ViewW_EdgeMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(eyesm);
   w->add(ptsm);
   w->add(resizem);
   w->add(cylsm);

   SetEpsilon(widget_scale*1e-4);
   
   FinishWidget(w);
}


ViewWidget::~ViewWidget()
{
}


void
ViewWidget::widget_execute()
{
   ((GeomSphere*)geometries[ViewW_GeomEye])->move(variables[ViewW_Eye]->Get(),
						  1*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomForeEye])->move(variables[ViewW_ForeEye]->Get(),
						      1*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomBackEye])->move(variables[ViewW_BackEye]->Get(),
						      1*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomPointUL])->move(variables[ViewW_PointUL]->Get(),
						      1*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomPointUR])->move(variables[ViewW_PointUR]->Get(),
						      1*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomPointDR])->move(variables[ViewW_PointDR]->Get(),
						      1*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomPointDL])->move(variables[ViewW_PointDL]->Get(),
						      1*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomCornerUL])->move(variables[ViewW_PointUL]->Get(),
						       0.5*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomCornerUR])->move(variables[ViewW_PointUR]->Get(),
						       0.5*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomCornerDR])->move(variables[ViewW_PointDR]->Get(),
						       0.5*widget_scale);
   ((GeomSphere*)geometries[ViewW_GeomCornerDL])->move(variables[ViewW_PointDL]->Get(),
						       0.5*widget_scale);
   Point p(variables[ViewW_PointUL]->Get() + (variables[ViewW_PointUR]->Get()
					      - variables[ViewW_PointUL]->Get()) / 2.0);
   ((GeomCappedCylinder*)geometries[ViewW_GeomResizeU])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   p = variables[ViewW_PointDR]->Get() + (variables[ViewW_PointDL]->Get()
					  - variables[ViewW_PointDR]->Get()) / 2.0;
   ((GeomCappedCylinder*)geometries[ViewW_GeomResizeD])->move(p - (GetAxis2() * 0.6 * widget_scale),
							      p + (GetAxis2() * 0.6 * widget_scale),
							      0.75*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomShaft])->move(variables[ViewW_Eye]->Get(),
						     variables[ViewW_BackEye]->Get(),
						     0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomCylU])->move(variables[ViewW_PointUL]->Get(),
						     variables[ViewW_PointUR]->Get(),
						     0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomCylR])->move(variables[ViewW_PointUR]->Get(),
						     variables[ViewW_PointDR]->Get(),
						     0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomCylD])->move(variables[ViewW_PointDR]->Get(),
						     variables[ViewW_PointDL]->Get(),
						     0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomCylL])->move(variables[ViewW_PointDL]->Get(),
						     variables[ViewW_PointUL]->Get(),
						     0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomEdgeU])->move(GetUL(),
						      GetUR(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomEdgeR])->move(GetUR(),
						      GetDR(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomEdgeD])->move(GetDR(),
						      GetDL(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomEdgeL])->move(GetDL(),
						      GetUL(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomDiagUL])->move(GetUL(),
						       variables[ViewW_Eye]->Get(),
						       0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomDiagUR])->move(GetUR(),
						       variables[ViewW_Eye]->Get(),
						       0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomDiagDR])->move(GetDR(),
						       variables[ViewW_Eye]->Get(),
						       0.5*widget_scale);
   ((GeomCylinder*)geometries[ViewW_GeomDiagDL])->move(GetDL(),
						       variables[ViewW_Eye]->Get(),
						       0.5*widget_scale);

   ((DistanceConstraint*)constraints[ViewW_ConstULUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ViewW_ConstDRDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ViewW_ConstULDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ViewW_ConstDRUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[ViewW_ConstULDR])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);
   ((DistanceConstraint*)constraints[ViewW_ConstURDL])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);

   SetEpsilon(widget_scale*1e-4);

   Vector spvec1(variables[ViewW_PointUR]->Get() - variables[ViewW_PointUL]->Get());
   Vector spvec2(variables[ViewW_PointDL]->Get() - variables[ViewW_PointUL]->Get());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if ((geom == ViewW_PickResizeU) || (geom == ViewW_PickResizeD))
	    picks[geom]->set_principal(spvec2);
	 else if ((geom == ViewW_PickEye) || (geom == ViewW_PickForeEye) || (geom == ViewW_PickBackEye))
	    picks[geom]->set_principal(v);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
ViewWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			void* cbdata )
{
   Vector delt(delta);
   Real t;
   
   ((DistanceConstraint*)constraints[ViewW_ConstULUR])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[ViewW_ConstDRDL])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[ViewW_ConstULDL])->SetDefault(GetAxis2());
   ((DistanceConstraint*)constraints[ViewW_ConstDRUR])->SetDefault(GetAxis2());

   for (Index v=0; v<NumVars; v++)
      variables[v]->Reset();
   
   switch((int)cbdata){
   case ViewW_PickEye:
      variables[ViewW_Eye]->SetDelta(delta);
      break;
   case ViewW_PickForeEye:
      variables[ViewW_PointUL]->MoveDelta(delta);
      variables[ViewW_PointUR]->MoveDelta(delta);
      variables[ViewW_PointDR]->MoveDelta(delta);
      variables[ViewW_PointDL]->MoveDelta(delta);
      variables[ViewW_BackEye]->SetDelta(delta);
      break;
   case ViewW_PickBackEye:
      variables[ViewW_BackEye]->SetDelta(delta);
      break;
   case ViewW_PickSphUL:
      variables[ViewW_PointUL]->SetDelta(delta);
      break;
   case ViewW_PickSphUR:
      variables[ViewW_PointUR]->SetDelta(delta);
      break;
   case ViewW_PickSphDR:
      variables[ViewW_PointDR]->SetDelta(delta);
      break;
   case ViewW_PickSphDL:
      variables[ViewW_PointDL]->SetDelta(delta);
      break;
   case ViewW_PickResizeU:
      if (((variables[ViewW_PointUL]->Get()+delta)-variables[ViewW_PointDL]->Get()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[ViewW_PointDL]->Get() + delta.normal()*3.2*widget_scale)
		 - variables[ViewW_PointUL]->Get());
      }
      t = delt.length();
      variables[ViewW_PointDL]->MoveDelta(GetAxis1()*t/2.0);
      variables[ViewW_PointDR]->MoveDelta(-GetAxis1()*t/2.0);
      variables[ViewW_PointUL]->MoveDelta(delt+GetAxis1()*t/2.0);
      variables[ViewW_PointUR]->SetDelta(delt-GetAxis1()*t/2.0, Scheme5);
      break;
   case ViewW_PickResizeD:
      if (((variables[ViewW_PointDR]->Get()+delta)-variables[ViewW_PointUR]->Get()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[ViewW_PointUR]->Get() + delta.normal()*3.2*widget_scale)
		 - variables[ViewW_PointDR]->Get());
      }
      t = delt.length();
      variables[ViewW_PointUL]->MoveDelta(GetAxis1()*t/2.0);
      variables[ViewW_PointUR]->MoveDelta(-GetAxis1()*t/2.0);
      variables[ViewW_PointDR]->MoveDelta(delt+GetAxis1()*t/2.0);
      variables[ViewW_PointDL]->SetDelta(delt-GetAxis1()*t/2.0, Scheme5);
      break;
   case ViewW_PickCyls:
      variables[ViewW_PointUL]->MoveDelta(delta);
      variables[ViewW_PointUR]->MoveDelta(delta);
      variables[ViewW_PointDR]->MoveDelta(delta);
      variables[ViewW_PointDL]->MoveDelta(delta);
      variables[ViewW_Eye]->MoveDelta(delta);
      variables[ViewW_ForeEye]->MoveDelta(delta);
      variables[ViewW_BackEye]->MoveDelta(delta);
      break;
   }
}

