/*
 * This code just spits out the sources for the
 * given amoeba data, or based on the widgets...
 *
 *
 *  Written by:
 *   Peter-Pike Sloan
 *   Department of Computer Science
 *   University of Utah
 *   April 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/KludgeMessage.h>
#include <Datatypes/KludgeMessagePort.h>
#include <Datatypes/SurfacePort.h>

#include <Datatypes/BasicSurfaces.h>

#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Widgets/PointWidget.h>
#include <TCL/TCLvar.h>

class DipoleCreate : public Module {
  AmoebaMessageIPort   *inamoeba;

  SurfaceOPort         *outp0;
  SurfaceOPort         *outp1;

  KludgeMessageOPort    *outsrc; // send something...

  GeometryOPort* ogeom;

  CrowdMonitor widget_lock0;
  PointWidget *widget0;

  CrowdMonitor widget_lock1; // just for orientation...
  PointWidget *widget1;
 

  int init;
public:
  DipoleCreate(const clString &id);
  DipoleCreate(const DipoleCreate&, int deep);

  virtual ~DipoleCreate();
  virtual Module* clone(int deep);
  virtual void execute();

  virtual void widget_moved(int lat);


};

extern "C" {
  Module* make_DipoleCreate(const clString& id)
    {
      return scinew DipoleCreate(id);
    }
};

DipoleCreate::DipoleCreate(const clString &id)
  :Module("DipoleCreate",id,Filter),init(0)
{
  // build input ports
  inamoeba = scinew AmoebaMessageIPort(this,"Amoeba Input",AmoebaMessageIPort::Atomic);
  add_iport(inamoeba);

  // build output ports

  outp0 = scinew SurfaceOPort(this,"Point 0",SurfaceIPort::Atomic);
  add_oport(outp0);
  
  outp1 = scinew SurfaceOPort(this,"Point 1",SurfaceIPort::Atomic);
  add_oport(outp1);

  ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);

  outsrc = scinew KludgeMessageOPort(this,"Source",KludgeMessageIPort::Atomic);
  add_oport(outsrc);

}


DipoleCreate::DipoleCreate(const DipoleCreate& copy, int deep)
: Module(copy, deep)
{
  NOT_FINISHED("DipoleCreate::DipoleCreate");
}

DipoleCreate::~DipoleCreate()
{
}

Module* DipoleCreate::clone(int deep)
{
  return scinew DipoleCreate(*this, deep);
}

void DipoleCreate::execute()
{
  AmoebaMessageHandle amh=0;

  int val=0;

  val = inamoeba->get(amh);

  if (val && !amh.get_rep()) {
    cerr << "connected, nothing there!\n";
    return;
  }


  if (!init) {
    init = 1;

    Point loc(75,170,90);
    Vector offset(10,15,12);

    //cerr << "We are here!\n";

    widget0 = scinew PointWidget(this,&widget_lock0,1.0);
    GeomObj *w = widget0->GetWidget();
    ogeom->addObj(w,"Dipole Location",&widget_lock0);
    widget0->Connect(ogeom);

    widget1 = scinew PointWidget(this,&widget_lock1,1.0);
    w = widget1->GetWidget();
    ogeom->addObj(w,"Dipole Location",&widget_lock1);
    widget1->Connect(ogeom);

    widget0->SetPosition(loc);
    widget1->SetPosition(loc+offset);

    widget0->SetScale(15);
    widget1->SetScale(10);

    ogeom->flushViews();
  }

  Point loc(80,140,80),loc1;
  Vector offset(10,15,12);
  
  double mag=1000;

  if (val && amh.get_rep()) {
    cerr << "We have a message!\n";
    if (amh->amoebas.size() == 1) {
      cerr << "We used it!\n";
      loc = amh->amoebas[0].sources[0].loc;
      double theta = amh->amoebas[0].sources[0].theta*2*M_PI;
      double phi = amh->amoebas[0].sources[0].phi*2*M_PI;
      
      double sinphi = sin(phi);
      
      offset = Vector(cos(theta)*sinphi,
		      sin(theta)*sinphi,
		      cos(phi))*offset.length();
      
      mag = amh->amoebas[0].sources[0].v;
      
      widget0->SetPosition(loc);
      widget1->SetPosition(loc+offset);
    }
  }

  loc = widget0->ReferencePoint();
  loc1 = widget1->ReferencePoint();

  offset = (loc1-loc).normal()*4; // fix this distance...

  //  cerr << (loc+offset) << " " << (loc-offset) << " " << offset << endl;

  // create the sources...

  SurfaceHandle s0 = scinew PointSurface(loc+offset);
  SurfaceHandle s1 = scinew PointSurface(loc-offset);

  s0->set_bc(to_string(mag));
  s1->set_bc(to_string(-mag));

  outp0->send(s0);
  outp1->send(s1);

  KludgeMessage *nkl = scinew KludgeMessage();

  nkl->src_recs.resize(1);

  Vector dirv = offset.normal();

  nkl->src_recs[0].loc = loc;
  nkl->src_recs[0].theta = atan2(dirv.y(),dirv.x())+2*M_PI;
  nkl->src_recs[0].phi = acos(dirv.z())/M_PI;

  nkl->src_recs[0].v = 1000;

  outsrc->send(KludgeMessageHandle(nkl));

}

void DipoleCreate::widget_moved(int last)
{
  if (last) {
    Point pt = widget0->ReferencePoint();

    cerr << pt << endl;
    
    
    //want_to_execute();
    
  }

}





