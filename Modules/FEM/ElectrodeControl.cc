/*
 * Code to sample "electrodes" from a scalar
 * Field - outputs stuff for them amoebers...
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/KludgeMessage.h>
#include <Datatypes/KludgeMessagePort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Malloc/Allocator.h>
#include <Datatypes/ScalarFieldUG.h>

#include <Multitask/ITC.h>
#include <Multitask/Task.h>

#include <limits.h>
#include <unistd.h>

#include <Datatypes/GeometryPort.h>
#include <Geom/Pt.h>

class WakeMeUp;

class ElectrodeControl : public Module {

  friend class WakeMeUp;

  ScalarFieldIPort   *infield;

  KludgeMessageOPort *outmessage;

  GeometryOPort      *ogeom;

  int init;

  Mutex DoneIt; // lock for have_executed...
  int have_executed;

public:
  ElectrodeControl(const clString &id);
  ElectrodeControl(const ElectrodeControl&, int deep);

  virtual ~ElectrodeControl();
  virtual Module* clone(int deep);
  virtual void execute();

};

extern "C" {
  Module* make_ElectrodeControl(const clString& id)
    {
      return scinew ElectrodeControl(id);
    }
};

ElectrodeControl::ElectrodeControl(const clString &id)
  :Module("ElectrodeControl",id,Filter),init(0),have_executed(0)
{
  infield = scinew ScalarFieldIPort(this,"Input Field",
				    ScalarFieldIPort::Atomic);
  add_iport(infield);

  outmessage = scinew KludgeMessageOPort(this,"Output Message",
					 KludgeMessageIPort::Atomic);
  add_oport(outmessage);

  ogeom = scinew GeometryOPort(this,"Geom",GeometryIPort::Atomic);
  add_oport(ogeom);
}


ElectrodeControl::ElectrodeControl(const ElectrodeControl& copy, int deep)
: Module(copy, deep)
{
  NOT_FINISHED("ElectrodeControl::ElectrodeControl");
}

ElectrodeControl::~ElectrodeControl()
{
}

Module* ElectrodeControl::clone(int deep)
{
  return scinew ElectrodeControl(*this, deep);
}

class WakeMeUp : public Task {
public:
  ElectrodeControl *owner;
  virtual int body(int);
  WakeMeUp(ElectrodeControl *);
};


WakeMeUp::WakeMeUp(ElectrodeControl *o)
  :Task("WakeUp Call"),owner(o)
{
  
}

int WakeMeUp::body(int)
{
#ifdef __sgi
  long sleep_ticks = 5*1000.0/CLK_TCK; // 30 seconds?

  // sleep for 20 seconds first time...

  cerr << "Taking 20 second nap!\n";

  sginap(100*1000.0/CLK_TCK); 

  cerr << "AAAAHHHH, Woke up!\n";

  while(1) {
    sginap(sleep_ticks);
    owner->DoneIt.lock();
    if (owner->have_executed)
      owner->want_to_execute();
    owner->have_executed = 0; // clear it out...
    owner->DoneIt.unlock();
  }
#endif
}


void ElectrodeControl::execute()
{
  ScalarFieldHandle inf;

  if (!infield->get(inf))
    return;

  static ScalarFieldUG *sf = 0;

  if (!inf->getUG())
    return;

  // see if this is a new field
  
  if (sf != inf->getUG()) { // yup, evaluate electrodes...

    sf = inf->getUG();

    // ok - we got everything...
    
    Array1<int> bdry_pts;
    
    cerr << "In ElectrodeControl\n";
    
    sf->mesh->get_boundary_nodes(bdry_pts);
#if 0
    GeomPts *cpts = scinew GeomPts(0);
    
    for(int ii=0;ii<bdry_pts.size();ii++) {
      cpts->add(sf->mesh->nodes[bdry_pts[ii]]->p);
    }
    
    ogeom->addObj(cpts,"Boundary Points");
    
    //return;
#endif
    
    // let's just take 128 random points for now...
    
    // now compute the offset...
    
    double offset=0.0;
    double berr=0.0;
    for(int i=0;i<bdry_pts.size();i++) {
      offset += sf->data[bdry_pts[i]];
    }
    
    offset /= bdry_pts.size();
    
    for(i=0;i<bdry_pts.size();i++) {
      //sf->data[bdry_pts[i]] -= offset;
      berr +=  (sf->data[bdry_pts[i]] - offset)*
	(sf->data[bdry_pts[i]] - offset);
    }
    
    cerr << berr << endl;
    
    cerr << offset << endl << endl << endl;
    
    KludgeMessage *kl = scinew KludgeMessage;
    
#if 0
    kl->surf_pots.resize(128);
    kl->surf_pti.resize(128);
    
    if (bdry_pts.size() <= 128) {
      cerr << "Not enough boundary points!\n";
      return;
    }
    
    for(int j=0;j<128;j++) {
      int index = bdry_pts.size()*drand48();
      while(bdry_pts[index] == -1)
	index = bdry_pts.size()*drand48();
      
      kl->surf_pti[j] = bdry_pts[index];
      kl->surf_pots[j] =sf->data[bdry_pts[index]] - offset; 
      
      //cerr << kl->surf_pti[j] << " " << kl->surf_pots[j] << endl;
      
      bdry_pts[index] = -1;
    }
#else
    kl->surf_pti = bdry_pts;
    kl->surf_pots.resize(bdry_pts.size());
    for(i=0;i<kl->surf_pti.size();i++) {
      kl->surf_pots[i] = sf->data[kl->surf_pti[i]]-offset;
    }
#endif
    outmessage->send(kl);
  } else { // send a "fire" message...
    KludgeMessage *kl = scinew KludgeMessage;

    // this will force a update

    outmessage->send(KludgeMessageHandle(kl));
  }
#if 1
  if (!init) {
    init = 1;
    cerr << "Creating this thread!\n";
    WakeMeUp *cur = scinew WakeMeUp(this);
    cur->activate(1);
  }
#endif
  DoneIt.lock();
  have_executed = 1;
  DoneIt.unlock();
}

