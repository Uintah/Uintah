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
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Widgets/ArrowWidget.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Switch.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <iostream>

#include <Packages/BioPSE/share/share.h>

using std::cerr;

namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE ShowDipoles : public Module {
  FieldIPort *ifield_;
  FieldOPort *ofield_;
  GeometryOPort* ogeom_;
  int gen_;
  FieldHandle dipoleFldH_;
  GuiString widgetSizeGui_;
  GuiString scaleModeGui_;
  GuiInt showLastVecGui_;
  GuiInt showLinesGui_;
  double lastSize_;
  clString execMsg_;
  Array1<GeomSwitch *> widget_switch_;
public:
  ShowDipoles(const clString& id);
  virtual ~ShowDipoles();
  virtual void execute();
  CrowdMonitor widget_lock_;
  Array1<int> widget_id_;
  Array1<ArrowWidget*> widget_;
  int gidx_;

  MaterialHandle greenMatl_;
  MaterialHandle deflMatl_;

  virtual void widget_moved(int last);
  int nDips_;
};

extern "C" BioPSESHARE Module* make_ShowDipoles(const clString& id) {
  return scinew ShowDipoles(id);
}

ShowDipoles::ShowDipoles(const clString& id) :
  Module("ShowDipoles", id, Filter, "Visualization", "BioPSE"),
  widgetSizeGui_("widgetSizeGui_", id, this),
  scaleModeGui_("scaleModeGui_", id, this),
  showLastVecGui_("showLastVecGui_", id, this),
  showLinesGui_("showLinesGui_", id, this),
  widget_lock_("ShowDipoles widget lock")
{
  // Create the input port
  ifield_=scinew FieldIPort(this, "dipoleFld", FieldIPort::Atomic);
  add_iport(ifield_);

  // Create the output ports
  ofield_=scinew FieldOPort(this, "dipoleFld", FieldIPort::Atomic);
  add_oport(ofield_);
  ogeom_=scinew GeometryOPort(this,"Geometry",GeometryIPort::Atomic);
  add_oport(ogeom_);
  gen_=-1;
  nDips_=0;
  lastSize_=-1;
  greenMatl_=new Material(Color(0.2, 0.8, 0.2));
  gidx_=0;
}

ShowDipoles::~ShowDipoles(){
}

void ShowDipoles::execute(){
  FieldHandle fieldH;
  PointCloud<Vector> *field_pcv;
  if (!ifield_->get(fieldH) || 
      !(field_pcv=dynamic_cast<PointCloud<Vector>*>(fieldH.get_rep()))) {
    cerr << "No vald input in ShowDipoles Field port.\n";
    return;
  }
  PointCloudMeshHandle field_mesh = field_pcv->get_typed_mesh();
  
  double widgetSize;
  if (!widgetSizeGui_.get().get_double(widgetSize)) {
    widgetSize=1;
    widgetSizeGui_.set("1.0");
  }
     
  if (fieldH->generation != gen_ || lastSize_ != widgetSize) {// load this data in
    if (field_pcv->fdata().size() != nDips_) {
	     
      cerr << "NEW SIZE FOR DIPOLEMATTOGEOM_  field_pcv->data().size()="<<field_pcv->fdata().size()<<" nDips_="<<nDips_<<"\n";
	     
      // nDips_ always just says how many switches we have set to true
      // need to fix switch setting first and then do allocations if
      //   necessary
	     
      if (widget_switch_.size()) {
	widget_[nDips_-1]->SetCurrentMode(0);
	widget_[nDips_-1]->SetMaterial(0, deflMatl_);
      }
      if (field_pcv->fdata().size()<nDips_) {
	for (int i=field_pcv->fdata().size(); i<nDips_; i++)
	  widget_switch_[i]->set_state(0);
	nDips_=field_pcv->fdata().size();
      } else {
	int i;
	for (i=nDips_; i<widget_switch_.size(); i++)
	  widget_switch_[i]->set_state(1);
	for (; i<field_pcv->fdata().size(); i++) {
	  widget_.add(scinew ArrowWidget(this, &widget_lock_, widgetSize));
	  deflMatl_=widget_[0]->GetMaterial(0);
	  widget_switch_.add(widget_[i]->GetWidget());
	  widget_switch_[i]->set_state(1);
	  widget_id_.add(ogeom_->addObj(widget_switch_[i], clString(clString("Dipole")+to_string(i)), &widget_lock_));
	}
	nDips_=field_pcv->fdata().size();
      }
      if (showLastVecGui_.get()) {
	widget_[nDips_-1]->SetCurrentMode(0);
	widget_[nDips_-1]->SetMaterial(0, deflMatl_);
      } else {
	widget_[nDips_-1]->SetCurrentMode(2);
	widget_[nDips_-1]->SetMaterial(0, greenMatl_);
      }
    }
    Array1<Point> pts;
    int i;
    clString scaleMode=scaleModeGui_.get();
    double max;
    for (i=0; i<field_pcv->fdata().size(); i++) {
      double dv=field_pcv->fdata()[i].length();
      if (dv<0.00000001) dv=1;
      if (i==0 || dv<max) max=dv;
    }

    for (i=0; i<field_pcv->fdata().size(); i++) {
      Point p;
      field_mesh->get_point(p,i);
      pts.add(p);
      widget_[i]->SetPosition(p);
      Vector v(field_pcv->fdata()[i]);
      //	     cerr << "widget_["<<i<<"] is at position "<<p<<" and dir "<<v<<"\n";
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

    if (gidx_) ogeom_->delObj(gidx_);
    if (showLinesGui_.get()) {
      GeomLines *g=new GeomLines;
      for (i=0; i<pts.size()-2; i++) 
	for (int j=i+1; j<pts.size()-1; j++) 
	  g->add(pts[i], pts[j]);
      GeomMaterial *gm=new GeomMaterial(g, new Material(Color(.8,.8,.2)));
      gidx_=ogeom_->addObj(gm, clString("Dipole Lines"));
    }

    gen_=fieldH->generation;
    fieldH.detach();
    dipoleFldH_=fieldH;
    lastSize_=widgetSize;
    ogeom_->flushViews();
    ofield_->send(dipoleFldH_);
    //     } else if (execMsg_ == "widget_moved") {
    //	 cerr << "Can't handle widget_moved callbacks yet...\n";
  } else if (execMsg_ == "widget_moved") {
    execMsg_="";
    Array1<Point> pts;
    int i;
    for (i=0; i<nDips_; i++) {
      Point p=widget_[i]->GetPosition();
      pts.add(p);
      Vector d=widget_[i]->GetDirection();
      double mag=widget_[i]->GetScale();
      cerr << "mag="<<mag<<"  widgetSize="<<widgetSize<<"\n";
      d=d*(mag/widgetSize);
      field_mesh->set_point(p, i);
      field_pcv->fdata()[i] = d;
    }
    ogeom_->delObj(gidx_);
    if (showLinesGui_.get()) {
      GeomLines *g=new GeomLines;
      for (i=0; i<pts.size()-2; i++) 
	for (int j=i+1; j<pts.size()-1; j++) 
	  g->add(pts[i], pts[j]);
      GeomMaterial *gm=new GeomMaterial(g, new Material(Color(.8,.8,.2)));
      gidx_=ogeom_->addObj(gm, clString("Dipole Lines"));
    }
    ogeom_->flushViews();
    dipoleFldH_=fieldH;
    ofield_->send(dipoleFldH_);
  } else {
    // just send the same old matrix/vector as last time
    cerr << "sending old stuff!\n";
    ofield_->send(dipoleFldH_);
  }
}

void ShowDipoles::widget_moved(int last) {
  if(last && !abort_flag) {
    abort_flag=1;
    execMsg_="widget_moved";
    want_to_execute();
  }
} 
} // End namespace BioPSE


