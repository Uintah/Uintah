/*
 *  ShowDipoles.cc:  Builds the RHS of the FE matrix for current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Widgets/ArrowWidget.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <iostream>

#include <Packages/BioPSE/share/share.h>

namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE ShowDipoles : public Module {
  FieldIPort *ifield_;
  FieldOPort *ofield_;
  GeometryOPort* ogeom_;

  FieldHandle dipoleFldH_;
  GuiDouble widgetSizeGui_;
  GuiString scaleModeGui_;
  GuiInt showLastVecGui_;
  GuiInt showLinesGui_;
  int lastGen_;
  double lastWidgetSize_;
  string lastScaleMode_;
  int lastShowLastVec_;
  int lastShowLines_;

  string execMsg_;
  Array1<GeomHandle> widget_switch_;
public:
  ShowDipoles(GuiContext *context);
  virtual ~ShowDipoles();
  virtual void execute();
  CrowdMonitor widget_lock_;
  Array1<int> widget_id_;
  Array1<ArrowWidget*> widget_;
  int gidx_;

  MaterialHandle greenMatl_;
  MaterialHandle deflMatl_;

  virtual void widget_moved(bool last);
  unsigned int nDips_;
};


DECLARE_MAKER(ShowDipoles)


ShowDipoles::ShowDipoles(GuiContext *context) :
  Module("ShowDipoles", context, Filter, "Visualization", "BioPSE"),
  widgetSizeGui_(context->subVar("widgetSizeGui_")),
  scaleModeGui_(context->subVar("scaleModeGui_")),
  showLastVecGui_(context->subVar("showLastVecGui_")),
  showLinesGui_(context->subVar("showLinesGui_")),
  widget_lock_("ShowDipoles widget lock")
{
  lastGen_=-1;
  lastWidgetSize_=-1;
  lastScaleMode_="";
  lastShowLastVec_=-1;
  lastShowLines_=-1;

  nDips_=0;
  greenMatl_=new Material(Color(0.2, 0.8, 0.2));
  gidx_=0;
}

ShowDipoles::~ShowDipoles(){
}


void
ShowDipoles::execute()
{
  ifield_ = (FieldIPort *)get_iport("dipoleFld");
  ofield_ = (FieldOPort *)get_oport("dipoleFld");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");

  if (!ifield_) {
    error("Unable to initialize iport 'dipoleFld'.");
    return;
  }
  if (!ofield_) {
    error("Unable to initialize oport 'dipoleFld'.");
    return;
  }
  if (!ogeom_) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }
  
  
  FieldHandle fieldH;
  PointCloudField<Vector> *field_pcv;
  if (!ifield_->get(fieldH) || 
      !(field_pcv=dynamic_cast<PointCloudField<Vector>*>(fieldH.get_rep()))) {
    error("No vald input in ShowDipoles Field port.");
    return;
  }
  PointCloudMeshHandle field_mesh = field_pcv->get_typed_mesh();
  
  double widgetSize = widgetSizeGui_.get();
  string scaleMode = scaleModeGui_.get();
  int showLastVec = showLastVecGui_.get();
  int showLines = showLinesGui_.get();
  int gen = fieldH->generation;
  
  if (gen != lastGen_ ||
      widgetSize != lastWidgetSize_ ||
      scaleMode != lastScaleMode_ ||
      showLastVec != lastShowLastVec_ ||
      showLines != lastShowLines_ ) {

    lastGen_ = fieldH->generation;
    if (field_pcv->fdata().size() != nDips_) {
	     
      msgStream_<< "NEW SIZE FOR DIPOLEMATTOGEOM_ field_pcv->data().size()="
		<< field_pcv->fdata().size() << " nDips_=" << nDips_ << "\n";
	     
      // nDips_ always just says how many switches we have set to true
      // need to fix switch setting first and then do allocations if
      //   necessary
	     
      if (widget_switch_.size()) {
	widget_[nDips_-1]->SetCurrentMode(1);
	widget_[nDips_-1]->SetMaterial(0, deflMatl_);
      }
      if (field_pcv->fdata().size()<nDips_) {
	for (unsigned int i=field_pcv->fdata().size(); i<nDips_; i++)
	  ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(0);
	nDips_=field_pcv->fdata().size();
      } else {
	unsigned int i;
	for (i=nDips_; i<(unsigned int)(widget_switch_.size()); i++)
	  ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(1);
	for (; i<field_pcv->fdata().size(); i++) {
	  ArrowWidget *a = scinew ArrowWidget(this, &widget_lock_, widgetSize);
	  a->Connect(ogeom_);
	  widget_.add(a);
	  deflMatl_=widget_[0]->GetMaterial(0);
	  widget_switch_.add(widget_[i]->GetWidget());
	  ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(1);
	  widget_id_.add(ogeom_->addObj(widget_switch_[i],
					"Dipole" + to_string((int)i),
					&widget_lock_));
	}
	nDips_=field_pcv->fdata().size();
      }
    }
    if (showLastVecGui_.get()) {
      widget_[nDips_-1]->SetCurrentMode(1);
      widget_[nDips_-1]->SetMaterial(0, deflMatl_);
    } else {
      widget_[nDips_-1]->SetCurrentMode(2);
      widget_[nDips_-1]->SetMaterial(0, greenMatl_);
    }
    Array1<Point> pts;
    unsigned int i;
    double max=0;
    for (i=0; i<field_pcv->fdata().size(); i++) {
      double dv=field_pcv->fdata()[i].length();
      if (i==0 || dv>max) max=dv;
    }

    for (i=0; i<field_pcv->fdata().size(); i++) {
      Point p;
      field_mesh->get_point(p,i);
      pts.add(p);
      widget_[i]->SetPosition(p);
      Vector v(field_pcv->fdata()[i]);
      double str=v.length();
      if (str<0.0000001) v.z(1);
      v.normalize();
      widget_[i]->SetDirection(v);
      //	     widget_[i]->SetScale(str*widgetSize);
      //	     widget_[i]->SetScale(widgetSize);
      double sc=widgetSize;
      if (scaleMode == "normalize") sc*=(str/max);
      else if (scaleMode == "scale") sc*=str;
      widget_[i]->SetScale(sc);
      widget_[i]->SetLength(2*sc);
    }

    if (gidx_) { ogeom_->delObj(gidx_); gidx_=0; }
    if (showLines) {
      GeomLines *g=new GeomLines;
      for (int i=0; i<pts.size()-1; i++) 
	for (int j=i+1; j<pts.size(); j++) 
	  g->add(pts[i], pts[j]);
      GeomMaterial *gm=new GeomMaterial(g, new Material(Color(.8,.8,.2)));
      gidx_=ogeom_->addObj(gm, string("Dipole Lines"));
    }
    FieldHandle fH(fieldH);
    fH.detach();
    dipoleFldH_=fH;
    ogeom_->flushViews();
    ofield_->send(dipoleFldH_);
  } else if (execMsg_ == "widget_moved") {
    execMsg_="";
    Array1<Point> pts;
    unsigned int i;
    FieldHandle fH(fieldH);
    fH.detach();
    fH->mesh_detach();
    field_pcv=dynamic_cast<PointCloudField<Vector>*>(fH.get_rep());
    field_mesh=field_pcv->get_typed_mesh();
    for (i=0; i<nDips_; i++) {
      Point p=widget_[i]->GetPosition();
      pts.add(p);
      Vector d=widget_[i]->GetDirection();
      double mag=widget_[i]->GetScale();
      d=d*(mag/widgetSize);
      field_mesh->set_point(p, i);
      field_pcv->fdata()[i] = d;
    }
    if (showLines) {
      GeomLines *g=new GeomLines;
      for (int i=0; i<pts.size()-2; i++) 
	for (int j=i+1; j<pts.size()-1; j++) 
	  g->add(pts[i], pts[j]);
      GeomMaterial *gm=new GeomMaterial(g, new Material(Color(.8,.8,.2)));
      ogeom_->delObj(gidx_);
      gidx_=ogeom_->addObj(gm, string("Dipole Lines"));
    }
    ogeom_->flushViews();
    dipoleFldH_=fH;
    ofield_->send(dipoleFldH_);
  } else {
    // just send the same old dipoles as last time
    remark("Sending old stuff.");
    ofield_->send(dipoleFldH_);
  }
  lastWidgetSize_ = widgetSize;
  lastScaleMode_ = scaleMode;
  lastShowLastVec_ = showLastVec;
  lastShowLines_ = showLines;
}


void
ShowDipoles::widget_moved(bool last)
{
  if(last && !abort_flag)
  {
    abort_flag=1;
    execMsg_="widget_moved";
    want_to_execute();
  }
}

 
} // End namespace BioPSE
