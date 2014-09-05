/*
 *  DipoleMatToGeom.cc:  Builds the RHS of the FE matrix for current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <PSECore/Widgets/ArrowWidget.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Switch.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace DaveW {
namespace Modules {
  
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;
using namespace SCICore::GeomSpace;

class DipoleMatToGeom : public Module {
  MatrixIPort *imat;
  MatrixOPort *omat;
  ColumnMatrixIPort *icol;
  ColumnMatrixOPort *ocol;
  GeometryOPort* ogeom;
  int gen;
  MatrixHandle dipoleMatH;
  TCLstring widgetSizeTCL;
  TCLstring scaleModeTCL;
  TCLint showLastVecTCL;
  TCLint showLinesTCL;
  int which;
  double lastSize;
  clString execMsg;
  Array1<GeomSwitch *> widget_switch;
public:
  DipoleMatToGeom(const clString& id);
  virtual ~DipoleMatToGeom();
  virtual void execute();
  CrowdMonitor widget_lock;
  Array1<int> widget_id;
  Array1<ArrowWidget*> widget;
  int gidx;

  MaterialHandle greenMatl;
  MaterialHandle deflMatl;

  virtual void widget_moved(int last);
  int nDips;
};

extern "C" Module* make_DipoleMatToGeom(const clString& id)
{
  return scinew DipoleMatToGeom(id);
}

DipoleMatToGeom::DipoleMatToGeom(const clString& id) : 
  Module("DipoleMatToGeom", id, Filter), 
  widgetSizeTCL("widgetSizeTCL", id, this),
  widget_lock("DipoleMatToGeom widget lock"),
  scaleModeTCL("scaleModeTCL", id, this),
  showLastVecTCL("showLastVecTCL", id, this),
  showLinesTCL("showLinesTCL", id, this)
{
  // Create the input port
  imat=scinew MatrixIPort(this, "DipoleMatrix", MatrixIPort::Atomic);
  add_iport(imat);

  // Create the output ports
  omat=scinew MatrixOPort(this, "DipoleMatrix", MatrixIPort::Atomic);
  add_oport(omat);
  ogeom=scinew GeometryOPort(this,"Geometry",GeometryIPort::Atomic);
  add_oport(ogeom);
  gen=-1;
  nDips=0;
  lastSize=-1;
  greenMatl=new Material(Color(0.2, 0.8, 0.2));
  gidx=0;
}

DipoleMatToGeom::~DipoleMatToGeom()
{
}

void DipoleMatToGeom::execute()
{
  MatrixHandle mh;
  Matrix* mp;
  if (!imat->get(mh) || !(mp=mh.get_rep())) {
    cerr << "No input in DipoleMatToGeom Matrix port.\n";
    return;
  }
  cerr << "nrows="<<mp->nrows()<<"  ncols="<<mp->ncols()<<"\n";
  if (mp->ncols() != 6) {
    cerr << "Error - dipoles must have six entries.\n";
    return;
  }
  double widgetSize;
  if (!widgetSizeTCL.get().get_double(widgetSize)) {
    widgetSize=1;
    widgetSizeTCL.set("1.0");
  }
     
  if (mh->generation != gen || lastSize != widgetSize) {// load this data in
    if (mp->nrows() != nDips) {
	     
      cerr << "NEW SIZE FOR DIPOLEMATTOGEOM  mp->nrows()="<<mp->nrows()<<" nDips="<<nDips<<"\n";
	     
      // nDips always just says how many switches we have set to true
      // need to fix switch setting first and then do allocations if
      //   necessary
	     
      if (widget_switch.size()) {
	widget[nDips-1]->SetCurrentMode(0);
	widget[nDips-1]->SetMaterial(0, deflMatl);
      }
      if (mp->nrows()<nDips) {
	for (int i=mp->nrows(); i<nDips; i++)
	  widget_switch[i]->set_state(0);
	nDips=mp->nrows();
      } else {
	int i;
	for (i=nDips; i<widget_switch.size(); i++)
	  widget_switch[i]->set_state(1);
	for (; i<mp->nrows(); i++) {
	  widget.add(scinew ArrowWidget(this, &widget_lock, widgetSize));
	  deflMatl=widget[0]->GetMaterial(0);
	  widget_switch.add(widget[i]->GetWidget());
	  widget_switch[i]->set_state(1);
	  widget_id.add(ogeom->addObj(widget_switch[i], clString(clString("Dipole")+to_string(i)), &widget_lock));
	}
	nDips=mp->nrows();
      }
      if (showLastVecTCL.get()) {
	widget[nDips-1]->SetCurrentMode(0);
	widget[nDips-1]->SetMaterial(0, deflMatl);
      } else {
	widget[nDips-1]->SetCurrentMode(2);
	widget[nDips-1]->SetMaterial(0, greenMatl);
      }
    }
    Array1<Point> pts;
    int i;
    clString scaleMode=scaleModeTCL.get();
    double max;
    for (i=0; i<mp->nrows(); i++) {
      double dv=Vector((*mp)[i][3], (*mp)[i][4], (*mp)[i][5]).length();
      if (dv<0.00000001) dv=1;
      if (i==0 || dv<max) max=dv;
    }

    for (i=0; i<mp->nrows(); i++) {
      Point p((*mp)[i][0], (*mp)[i][1], (*mp)[i][2]);
      pts.add(p);
      widget[i]->SetPosition(p);
      Vector v((*mp)[i][3], (*mp)[i][4], (*mp)[i][5]);
      //	     cerr << "widget["<<i<<"] is at position "<<p<<" and dir "<<v<<"\n";
      double str=v.length();
      if (str<0.0000001) v.z(1);
      v.normalize();
      widget[i]->SetDirection(v);
      //	     widget[i]->SetScale(str*widgetSize);
      //	     widget[i]->SetScale(widgetSize);
      double sc=widgetSize;
      if (scaleMode == "normalize") sc*=(str/max);
      else if (scaleMode == "scale") sc*=str;
      widget[i]->SetScale(sc);
      widget[i]->SetLength(2*sc);
    }

    if (gidx) ogeom->delObj(gidx);
    if (showLinesTCL.get()) {
      GeomLines *g=new GeomLines;
      for (i=0; i<pts.size()-2; i++) 
	for (int j=i+1; j<pts.size()-1; j++) 
	  g->add(pts[i], pts[j]);
      GeomMaterial *gm=new GeomMaterial(g, new Material(Color(.8,.8,.2)));
      gidx=ogeom->addObj(gm, clString("Dipole Lines"));
    }

    gen=mh->generation;
    dipoleMatH=mh;
    lastSize=widgetSize;
    ogeom->flushViews();
    omat->send(dipoleMatH);
    //     } else if (execMsg == "widget_moved") {
    //	 cerr << "Can't handle widget_moved callbacks yet...\n";
  } else if (execMsg == "widget_moved") {
    execMsg="";
    Array1<Point> pts;
    int i;
    for (i=0; i<nDips; i++) {
      Point p=widget[i]->GetPosition();
      pts.add(p);
      Vector d=widget[i]->GetDirection();
      double mag=widget[i]->GetScale();
      cerr << "mag="<<mag<<"  widgetSize="<<widgetSize<<"\n";
      d=d*(mag/widgetSize);
      (*mp)[i][0]=p.x();
      (*mp)[i][1]=p.y();
      (*mp)[i][2]=p.z();
      (*mp)[i][3]=d.x();
      (*mp)[i][4]=d.y();
      (*mp)[i][5]=d.z();
    }
    ogeom->delObj(gidx);
    if (showLinesTCL.get()) {
      GeomLines *g=new GeomLines;
      for (i=0; i<pts.size()-2; i++) 
	for (int j=i+1; j<pts.size()-1; j++) 
	  g->add(pts[i], pts[j]);
      GeomMaterial *gm=new GeomMaterial(g, new Material(Color(.8,.8,.2)));
      gidx=ogeom->addObj(gm, clString("Dipole Lines"));
    }
    ogeom->flushViews();
    dipoleMatH=mh;
    omat->send(dipoleMatH);
  } else {
    // just send the same old matrix/vector as last time
    cerr << "sending old stuff!\n";
    omat->send(dipoleMatH);
  }

  //     cerr << "DipoleMatToGeom: Here are the dipoles...\n";
  for (int i=0; i<mp->nrows(); i++) {
    //	 cerr << "   "<<i<<"   ";
    for (int j=0; j<mp->ncols(); j++) {
      //	     cerr << (*mp)[i][j]<<" ";
    }
    //	 cerr << "\n";
  }

}

void DipoleMatToGeom::widget_moved(int last) {
  if(last && !abort_flag) {
    abort_flag=1;
    execMsg="widget_moved";
    want_to_execute();
  }
}
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.7  2000/11/16 03:39:53  dmw
// added show lines flag
//
// Revision 1.6  2000/10/29 03:51:45  dmw
// SeedDipoles will place dipoles randomly within a mesh
//
// Revision 1.5  2000/08/01 18:03:03  dmw
// fixed errors
//
// Revision 1.4  2000/03/17 09:25:43  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  1999/10/07 02:06:34  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/09/08 02:26:27  sparker
// Various #include cleanups
//
// Revision 1.1  1999/09/02 04:49:24  dmw
// more of Dave's modules
//
//
