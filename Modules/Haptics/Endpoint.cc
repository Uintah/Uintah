 // WRITTEN OR MODIFIED BY LDURBECK ldurbeck 6/11/98

/* Endpoint.cc: first representation of phantom in scirun 
 *   
 *  
 * Written by Lisa Durbeck
 *
 * based on
 *  Simple "Template module"for the SCIRun                      *
 *  ~/zhukov/SciRun/Modules/Template.cc                         *
 *  Written by:                                                 *
 *   Leonid Zhukov                                              *
 *                                                              *
 ****************************************************************/

#include <Classlib/NotFinished.h>
#include <Datatypes/VoidStar.h>
#include <Datatypes/VoidStarPort.h>
#include <Datatypes/GeometryPort.h>
#include <Widgets/PointWidget.h>
#include <Dataflow/Module.h>
#include <Geom/BBoxCache.h>
#include <Geometry/BBox.h>
#include <Geom/Pt.h>

#include <TCL/TCLvar.h>
#include <iostream.h>
#include <stdio.h>


class Endpoint : public Module {
 
public:
  TCLstring tcl_status;
  Endpoint(const clString& id);
  Endpoint(const Endpoint&, int deep);
  virtual ~Endpoint();
  virtual Module* clone(int deep);
  virtual void execute();
  virtual void widget_moved(int); // from Dataflow/Module.h

private:
// I/O ports (not actually ports here: shared memory)
  GeometryOPort * gop;
  VoidStarIPort * input;
  PhantomXYZ * position;  // output position of phantom (in what coordinate system?)      
  PointWidget * point;
  
}; //class


extern "C" {
  
  Module* make_Endpoint(const clString& id)
  {
    return new Endpoint(id);
  }
  
};//extern


//---------------------------------------------------------------
Endpoint::Endpoint(const clString& id)
  : Module("Endpoint", id, Filter),
    tcl_status("tcl_status",id,this)

{
// PUT INITIALIZATION STUFF HERE
    input = new VoidStarIPort(this, "PhantomXYZ", VoidStarIPort::Atomic);
    add_iport(input);
    gop = new GeometryOPort(this, "Phantom Endpoint", GeometryIPort::Atomic);
    add_oport(gop);
// create a bounding box
    GeomPts * pp = new GeomPts(0);
    pp->add(Point(0.0, 0.0, 0.0));
    BBox b;
    b.extend(Point(-1,-1,-1)); // change this to enlarge/shrink box
    b.extend(Point(1,1,1));
    GeomBBoxCache * box = new GeomBBoxCache(pp, b);
    gop->addObj(box, "Bounding Box");
 
}

//----------------------------------------------------------
Endpoint::Endpoint(const Endpoint& copy, int deep)
  : Module(copy, deep),
    tcl_status("tcl_status",id,this)   
{}

//------------------------------------------------------------
Endpoint::~Endpoint(){}

//-------------------------------------------------------------
Module* Endpoint::clone(int deep)
{
  return new Endpoint(*this, deep);
}

//--------------------------------------------------------------

void Endpoint::execute()
{
  VoidStarHandle pHandle;
  input->get(pHandle);
  if (!pHandle.get_rep()) return;
  if (!(position = pHandle->getPhantomXYZ())) return;

  // now have input position to work with.

// get current position and create/move point to it
  position->updateLock.read_lock();  
  Point p1 = position->position.point();
  position->updateLock.read_unlock();
// create a point widget to send to salmon
  CrowdMonitor *l = new CrowdMonitor;
  point = new PointWidget(this,  l, 0.05);
  point->SetPosition(p1);
  GeomObj *w = point->GetWidget();
  gop->addObj(w, "Endpoint", l);
  gop->flushViews();

 while (1) {

  position->Esem.down();
  position->updateLock.read_lock();  
  p1 = position->position.point();
  position->updateLock.read_unlock();
  point->SetPosition(p1);
  gop->flushViews();
}  
} 


void Endpoint::widget_moved(int last) {

   // last is MouseUp flag. I don't use it
  bool changed = false; 
  Point p;
  p = point->GetPosition(); // read current position of sphere widget

// keep user within bounding box
  if (p.x() < -1) {
    p.x(-1);
    changed = true;
  }
  else if (p.x() > 1) {
    p.x(1);
    changed = true;
  }
  if (p.y() < -1) {
    p.y(-1);
    changed = true;
  }
  else if (p.y() > 1) {
    p.y(1);
    changed = true;
  }
  if (p.z() < -1) {
    p.z(-1);
    changed = true;
  }
  else if (p.z() > 1) {
    p.z(1);
    changed = true;
  }

if (changed) {
//  point->SetPosition(p);
}


// set shared position vector
       position->updateLock.write_lock();
       position->position = p.vector();
       position->updateLock.write_unlock();


}



//---------------------------------------------------------------











