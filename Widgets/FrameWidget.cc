
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
const Index NumGeoms = 12;
const Index NumMatls = 3;
const Index NumPcks = 9;
const Index NumSchemes = 4;

enum { FrameW_ConstULDR, FrameW_ConstURDL, FrameW_ConstPyth, FrameW_ConstPlane,
       FrameW_ConstULUR, FrameW_ConstULDL, FrameW_ConstDRUR, FrameW_ConstDRDL };
enum { FrameW_GeomPointUL, FrameW_GeomPointUR, FrameW_GeomPointDR, FrameW_GeomPointDL,
       FrameW_GeomCylU, FrameW_GeomCylR, FrameW_GeomCylD, FrameW_GeomCylL,
       FrameW_GeomResizeU, FrameW_GeomResizeR, FrameW_GeomResizeD, FrameW_GeomResizeL };
enum { FrameW_PickSphUL, FrameW_PickSphUR, FrameW_PickSphDR, FrameW_PickSphDL, FrameW_PickCyls,
       FrameW_PickResizeU, FrameW_PickResizeR, FrameW_PickResizeD, FrameW_PickResizeL };

FrameWidget::FrameWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1),
  oldaxis1(1, 0, 0), oldaxis2(1, 0, 0)
{
   Real INIT = 1.0*widget_scale;
   // Scheme2/3 are used by the picks in GeomMoved!!
   variables[FrameW_PointUL] = new Variable("PntUL", Scheme1, Point(0, 0, 0));
   variables[FrameW_PointUR] = new Variable("PntUR", Scheme2, Point(INIT, 0, 0));
   variables[FrameW_PointDR] = new Variable("PntDR", Scheme1, Point(INIT, INIT, 0));
   variables[FrameW_PointDL] = new Variable("PntDL", Scheme2, Point(0, INIT, 0));
   variables[FrameW_Dist1] = new Variable("DIST1", Scheme1, Point(INIT, 0, 0));
   variables[FrameW_Dist2] = new Variable("DIST2", Scheme1, Point(INIT, 0, 0));
   variables[FrameW_Hypo] = new Variable("HYPO", Scheme1, Point(sqrt(2*INIT*INIT), 0, 0));

   constraints[FrameW_ConstPlane] = new PlaneConstraint("ConstPlane",
							NumSchemes,
							variables[FrameW_PointUL],
							variables[FrameW_PointUR],
							variables[FrameW_PointDR],
							variables[FrameW_PointDL]);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme1, 2, 3, 0, 1);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 1);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme3, 2, 3, 0, 1);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme4, 2, 3, 0, 1);
   constraints[FrameW_ConstPlane]->Priorities(P_Highest, P_Highest,
					      P_Highest, P_Highest);
   constraints[FrameW_ConstULDR] = new DistanceConstraint("Const13",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointDR],
							  variables[FrameW_Hypo]);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme3, 2, 2, 1);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[FrameW_ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[FrameW_ConstURDL] = new DistanceConstraint("Const24",
							  NumSchemes,
							  variables[FrameW_PointUR],
							  variables[FrameW_PointDL],
							  variables[FrameW_Hypo]);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme2, 1, 0, 1);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme4, 2, 2, 1);
   constraints[FrameW_ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[FrameW_ConstPyth] = new PythagorasConstraint("ConstPyth",
							    NumSchemes,
							    variables[FrameW_Dist1],
							    variables[FrameW_Dist2],
							    variables[FrameW_Hypo]);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme2, 1, 0, 1);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme3, 2, 2, 0);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme4, 2, 2, 1);
   constraints[FrameW_ConstPyth]->Priorities(P_Default, P_Default, P_HighMedium);
   constraints[FrameW_ConstULUR] = new DistanceConstraint("Const12",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointUR],
							  variables[FrameW_Dist1]);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FrameW_ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstULDL] = new DistanceConstraint("Const14",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointDL],
							  variables[FrameW_Dist2]);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FrameW_ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstDRUR] = new DistanceConstraint("Const32",
							  NumSchemes,
							  variables[FrameW_PointDR],
							  variables[FrameW_PointUR],
							  variables[FrameW_Dist2]);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FrameW_ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstDRDL] = new DistanceConstraint("Const34",
							  NumSchemes,
							  variables[FrameW_PointDR],
							  variables[FrameW_PointDL],
							  variables[FrameW_Dist1]);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FrameW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[FrameW_PointMatl] = PointWidgetMaterial;
   materials[FrameW_EdgeMatl] = EdgeWidgetMaterial;
   materials[FrameW_HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = FrameW_GeomPointUL, pick = FrameW_PickSphUL;
	geom <= FrameW_GeomPointDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[FrameW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[FrameW_PointMatl]);
   
   GeomGroup* resizes = new GeomGroup;
   for (geom = FrameW_GeomResizeU, pick = FrameW_PickResizeU;
	geom <= FrameW_GeomResizeL; geom++, pick++) {
      geometries[geom] = new GeomCappedCylinder;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[FrameW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      resizes->add(picks[pick]);
   }
   GeomMaterial* resizem = new GeomMaterial(resizes, materials[FrameW_PointMatl]);

   GeomGroup* cyls = new GeomGroup;
   for (geom = FrameW_GeomCylU; geom <= FrameW_GeomCylL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[FrameW_PickCyls] = new GeomPick(cyls, module);
   picks[FrameW_PickCyls]->set_highlight(materials[FrameW_HighMatl]);
   picks[FrameW_PickCyls]->set_cbdata((void*)FrameW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[FrameW_PickCyls], materials[FrameW_EdgeMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(resizem);
   w->add(cylsm);

   SetEpsilon(widget_scale*1e-4);
   
   FinishWidget(w);
}


FrameWidget::~FrameWidget()
{
}


void
FrameWidget::widget_execute()
{
   ((GeomSphere*)geometries[FrameW_GeomPointUL])->move(variables[FrameW_PointUL]->Get(),
						       1*widget_scale);
   ((GeomSphere*)geometries[FrameW_GeomPointUR])->move(variables[FrameW_PointUR]->Get(),
						       1*widget_scale);
   ((GeomSphere*)geometries[FrameW_GeomPointDR])->move(variables[FrameW_PointDR]->Get(),
						       1*widget_scale);
   ((GeomSphere*)geometries[FrameW_GeomPointDL])->move(variables[FrameW_PointDL]->Get(),
						       1*widget_scale);
   Point p(variables[FrameW_PointUL]->Get() + (variables[FrameW_PointUR]->Get()
					       - variables[FrameW_PointUL]->Get()) / 2.0);
   ((GeomCappedCylinder*)geometries[FrameW_GeomResizeU])->move(p - (GetAxis2() * 0.6 * widget_scale),
							       p + (GetAxis2() * 0.6 * widget_scale),
							       0.75*widget_scale);
   p = variables[FrameW_PointUR]->Get() + (variables[FrameW_PointDR]->Get()
					   - variables[FrameW_PointUR]->Get()) / 2.0;
   ((GeomCappedCylinder*)geometries[FrameW_GeomResizeR])->move(p - (GetAxis1() * 0.6 * widget_scale),
							       p + (GetAxis1() * 0.6 * widget_scale),
							       0.75*widget_scale);
   p = variables[FrameW_PointDR]->Get() + (variables[FrameW_PointDL]->Get()
					   - variables[FrameW_PointDR]->Get()) / 2.0;
   ((GeomCappedCylinder*)geometries[FrameW_GeomResizeD])->move(p - (GetAxis2() * 0.6 * widget_scale),
							       p + (GetAxis2() * 0.6 * widget_scale),
							       0.75*widget_scale);
   p = variables[FrameW_PointDL]->Get() + (variables[FrameW_PointUL]->Get()
					   - variables[FrameW_PointDL]->Get()) / 2.0;
   ((GeomCappedCylinder*)geometries[FrameW_GeomResizeL])->move(p - (GetAxis1() * 0.6 * widget_scale),
							       p + (GetAxis1() * 0.6 * widget_scale),
							       0.75*widget_scale);
   ((GeomCylinder*)geometries[FrameW_GeomCylU])->move(variables[FrameW_PointUL]->Get(),
						      variables[FrameW_PointUR]->Get(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_GeomCylR])->move(variables[FrameW_PointUR]->Get(),
						      variables[FrameW_PointDR]->Get(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_GeomCylD])->move(variables[FrameW_PointDR]->Get(),
						      variables[FrameW_PointDL]->Get(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_GeomCylL])->move(variables[FrameW_PointDL]->Get(),
						      variables[FrameW_PointUL]->Get(),
						      0.5*widget_scale);

   ((DistanceConstraint*)constraints[FrameW_ConstULUR])->SetMinimum(2*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstDRDL])->SetMinimum(2*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstULDL])->SetMinimum(2*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstDRUR])->SetMinimum(2*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstULDR])->SetMinimum(sqrt(8)*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstURDL])->SetMinimum(sqrt(8)*widget_scale);

   SetEpsilon(widget_scale*1e-4);

   Vector spvec1(variables[FrameW_PointUR]->Get() - variables[FrameW_PointUL]->Get());
   Vector spvec2(variables[FrameW_PointDL]->Get() - variables[FrameW_PointUL]->Get());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if ((geom == FrameW_PickResizeU) || (geom == FrameW_PickResizeD))
	    picks[geom]->set_principal(spvec2);
	 else if ((geom == FrameW_PickResizeL) || (geom == FrameW_PickResizeR))
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
FrameWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   int pick((int)cbdata);
   
   // This axis code is to ensure that resizing through zero lengths works correctly.
   Vector axis1(variables[FrameW_PointUR]->Get() - variables[FrameW_PointUL]->Get());
   Vector axis2(variables[FrameW_PointDL]->Get() - variables[FrameW_PointUL]->Get());
   if (axis1.length2() > 1e-6) {
      // We've got an axis, so let's use it.
      axis1.normalize();
   } else if ((pick == FrameW_PickResizeL) || (pick == FrameW_PickResizeR)) {
      // Let's grow uniformly in the same direction.
      axis1 = delta;
      axis1.normalize();
   } else {
      // Let's use the saved value from GetAxis.
      axis1 = oldaxis1;
   }
   ((DistanceConstraint*)constraints[FrameW_ConstULUR])->SetDefault(axis1);
   ((DistanceConstraint*)constraints[FrameW_ConstDRDL])->SetDefault(axis1);
   
   if (axis2.length2() > 1e-6) {
      // We've got an axis, so let's use it.
      axis2.normalize();
   } else if ((pick == FrameW_PickResizeU) || (pick == FrameW_PickResizeD)) {
      // Let's grow uniformly in the same direction.
      axis2 = delta;
      axis2.normalize();
   } else {
      // Let's use the saved value from GetAxis.
      axis2 = oldaxis2;
   }
   ((DistanceConstraint*)constraints[FrameW_ConstULDL])->SetDefault(axis2);
   ((DistanceConstraint*)constraints[FrameW_ConstDRUR])->SetDefault(axis2);
   
   for (Index v=0; v<NumVars; v++)
      variables[v]->Reset();
   
   switch(pick){
   case FrameW_PickSphUL:
      variables[FrameW_PointUL]->SetDelta(delta);
      break;
   case FrameW_PickSphUR:
      variables[FrameW_PointUR]->SetDelta(delta);
      break;
   case FrameW_PickSphDR:
      variables[FrameW_PointDR]->SetDelta(delta);
      break;
   case FrameW_PickSphDL:
      variables[FrameW_PointDL]->SetDelta(delta);
      break;
   case FrameW_PickResizeU:
      variables[FrameW_PointUL]->MoveDelta(delta);
      variables[FrameW_PointUR]->SetDelta(delta, Scheme4);
      break;
   case FrameW_PickResizeR:
      variables[FrameW_PointUR]->MoveDelta(delta);
      variables[FrameW_PointDR]->SetDelta(delta, Scheme3);
      break;
   case FrameW_PickResizeD:
      variables[FrameW_PointDR]->MoveDelta(delta);
      variables[FrameW_PointDL]->SetDelta(delta, Scheme4);
      break;
   case FrameW_PickResizeL:
      variables[FrameW_PointDL]->MoveDelta(delta);
      variables[FrameW_PointUL]->SetDelta(delta, Scheme3);
      break;
   case FrameW_PickCyls:
      variables[FrameW_PointUL]->MoveDelta(delta);
      variables[FrameW_PointUR]->MoveDelta(delta);
      variables[FrameW_PointDR]->MoveDelta(delta);
      variables[FrameW_PointDL]->MoveDelta(delta);
      break;
   }
}

