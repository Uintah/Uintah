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
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Widgets/ArrowWidget.h>
#include <Core/Datatypes/DenseMatrix.h>
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
  MatrixIPort *imat_;
  MatrixOPort *omat_;
  GeometryOPort* ogeom_;
  int gen_;
  MatrixHandle dipoleMatH_;
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
  imat_=scinew MatrixIPort(this, "DipoleMatrix", MatrixIPort::Atomic);
  add_iport(imat_);

  // Create the output ports
  omat_=scinew MatrixOPort(this, "DipoleMatrix", MatrixIPort::Atomic);
  add_oport(omat_);
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
  MatrixHandle mh;
  Matrix* mp;
  if (!imat_->get(mh) || !(mp=mh.get_rep())) {
    cerr << "No input in ShowDipoles Matrix port.\n";
    return;
  }
  cerr << "nrows="<<mp->nrows()<<"  ncols="<<mp->ncols()<<"\n";
  if (mp->ncols() != 6) {
    cerr << "Error - dipoles must have six entries.\n";
    return;
  }
  double widgetSize;
  if (!widgetSizeGui_.get().get_double(widgetSize)) {
    widgetSize=1;
    widgetSizeGui_.set("1.0");
  }
     
  if (mh->generation != gen_ || lastSize_ != widgetSize) {// load this data in
    if (mp->nrows() != nDips_) {
	     
      cerr << "NEW SIZE FOR DIPOLEMATTOGEOM_  mp->nrows()="<<mp->nrows()<<" nDips_="<<nDips_<<"\n";
	     
      // nDips_ always just says how many switches we have set to true
      // need to fix switch setting first and then do allocations if
      //   necessary
	     
      if (widget_switch_.size()) {
	widget_[nDips_-1]->SetCurrentMode(0);
	widget_[nDips_-1]->SetMaterial(0, deflMatl_);
      }
      if (mp->nrows()<nDips_) {
	for (int i=mp->nrows(); i<nDips_; i++)
	  widget_switch_[i]->set_state(0);
	nDips_=mp->nrows();
      } else {
	int i;
	for (i=nDips_; i<widget_switch_.size(); i++)
	  widget_switch_[i]->set_state(1);
	for (; i<mp->nrows(); i++) {
	  widget_.add(scinew ArrowWidget(this, &widget_lock_, widgetSize));
	  deflMatl_=widget_[0]->GetMaterial(0);
	  widget_switch_.add(widget_[i]->GetWidget());
	  widget_switch_[i]->set_state(1);
	  widget_id_.add(ogeom_->addObj(widget_switch_[i], clString(clString("Dipole")+to_string(i)), &widget_lock_));
	}
	nDips_=mp->nrows();
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
    for (i=0; i<mp->nrows(); i++) {
      double dv=Vector((*mp)[i][3], (*mp)[i][4], (*mp)[i][5]).length();
      if (dv<0.00000001) dv=1;
      if (i==0 || dv<max) max=dv;
    }

    for (i=0; i<mp->nrows(); i++) {
      Point p((*mp)[i][0], (*mp)[i][1], (*mp)[i][2]);
      pts.add(p);
      widget_[i]->SetPosition(p);
      Vector v((*mp)[i][3], (*mp)[i][4], (*mp)[i][5]);
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

    gen_=mh->generation;
    dipoleMatH_=mh;
    lastSize_=widgetSize;
    ogeom_->flushViews();
    omat_->send(dipoleMatH_);
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
      (*mp)[i][0]=p.x();
      (*mp)[i][1]=p.y();
      (*mp)[i][2]=p.z();
      (*mp)[i][3]=d.x();
      (*mp)[i][4]=d.y();
      (*mp)[i][5]=d.z();
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
    dipoleMatH_=mh;
    omat_->send(dipoleMatH_);
  } else {
    // just send the same old matrix/vector as last time
    cerr << "sending old stuff!\n";
    omat_->send(dipoleMatH_);
  }

  //     cerr << "ShowDipoles: Here are the dipoles...\n";
  for (int i=0; i<mp->nrows(); i++) {
    //	 cerr << "   "<<i<<"   ";
    for (int j=0; j<mp->ncols(); j++) {
      //	     cerr << (*mp)[i][j]<<" ";
    }
    //	 cerr << "\n";
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


