 // WRITTEN OR MODIFIED BY LDURBECK ldurbeck 6/11/98

/* Force.cc: sends a force to the phantom. 
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
#include <Malloc/Allocator.h>
#include <Datatypes/VoidStar.h>
#include <Datatypes/VoidStarPort.h>
#include <Datatypes/Interval.h>
#include <Datatypes/IntervalPort.h>
#include <Dataflow/Module.h>
#include <TCL/TCLvar.h>
#ifdef SOCKETS
 #include <Modules/Haptics/client.c> 
 //#include <Datatypes/ljdGlobals.h>
#endif
#include <iostream.h>
#include <stdio.h>


class Force : public Module {
 
public:
  inline void setForce(double u, double v, double w) {
#ifdef SOCKETS // send to phantom
// scale from data range to full range of phantom forces
// ASSUMPTION: this code knows what the maximum force the phantom
// should exert is: LATER pass that value in from the phantom
// LATER make PhantomForceMax alterable by user but keep a cap/max value

  // (u,v,w) is from the data: scale it to the phantom before sending it.
// this means I'm going from the DRange coordinate system to the PRange system.


   double xc, yc, zc;
       // xc is x component, yc is y component, zc is z component of force
   
   if (DRangeX->low == DRangeX->high) {  // not an error
     cout << "NOT GETTING DATA RANGE...PROCEEDING WITHOUT IT" << endl;
     xc = Min(u, PRangeX->high); // can't scale -- just cap it off at a safe value 
   }
   else {
     xc = (u - DRangeX->low)/(DRangeX->high - DRangeX->low); // where u lies within DRange
     xc = xc * (PRangeX->high - PRangeX->low); // where u should lie within PRange
     xc = xc + PRangeX->low;  // starting at the right place
   }

   if (DRangeY->low == DRangeY->high) {  // not an error
     yc = Min(v, PRangeY->high); // can't scale -- just cap it off at a safe value 
   }
   else {
     yc = (v - DRangeY->low)/(DRangeY->high - DRangeY->low); // where v lies within DRange
     yc = yc * (PRangeY->high - PRangeY->low); // where v  should lie within PRange
     yc = yc + PRangeY->low;  // starting at the right place
   }

   if (DRangeZ->low == DRangeZ->high) { // not an error
     zc = Min(w, PRangeZ->high); // can't scale -- just cap it off at a safe value 
   }
   else {
     zc = (w - DRangeZ->low)/(DRangeZ->high - DRangeZ->low); // where z lies within DRange
     zc = zc * (PRangeZ->high - PRangeZ->low); // where z should lie within PRange
     zc = zc + PRangeZ->low;  // starting at the right place
   }

// INV: now I have a scaled force the phantom can use              
   //printf("(%g   %g   %g)\n", xc, yc, zc);
   write_triple(xc,yc,zc);
   //write_triple(0.0, 0.0, 0.0);

#else 
   //printf("got %g\t%g\t%g\n",u,v,w);
#endif 
  

}
 
  TCLstring tcl_status;
  Force(const clString& id);
  Force(const Force&, int deep);
  virtual ~Force();
  virtual Module* clone(int deep);
  virtual void execute();

private:
// I/O ports (not actually ports here: shared memory)
  VoidStarIPort * input;
  PhantomUVW * force;       

// regular old I/O ports and associated memory
  IntervalHandle DRangeX, DRangeY, DRangeZ; // data value ranges along x,y,and z
  IntervalHandle PRangeX, PRangeY, PRangeZ;  // phantom force ranges along x, y, and z
  IntervalIPort * rangeIport; // to get the intervals from PositionToForce module 
}; //class


extern "C" {
  
  Module* make_Force(const clString& id)
  {
    return new Force(id);
  }
  
};//extern


//---------------------------------------------------------------
Force::Force(const clString& id)
  : Module("Force", id, Filter),
    tcl_status("tcl_status",id,this)

{
// PUT INITIALIZATION STUFF HERE
    input = scinew VoidStarIPort(this, "PhantomUVW", VoidStarIPort::Atomic);
    add_iport(input);
    rangeIport = scinew IntervalIPort(this, "Force/Value Intervals", IntervalIPort::Atomic);
    add_iport(rangeIport); 
}

//----------------------------------------------------------
Force::Force(const Force& copy, int deep)
  : Module(copy, deep),
    tcl_status("tcl_status",id,this)   
{}

//------------------------------------------------------------
Force::~Force(){}

//-------------------------------------------------------------
Module* Force::clone(int deep)
{
  return new Force(*this, deep);
}

//--------------------------------------------------------------

void Force::execute()
{

// DO ACTUAL WORK HERE

// GET INITIAL DATA

// first I expect a batch of interval/range information so that I
// can convert from data space to phantom space successfully in setForce().
// ASSUMPTION: I'll get the ranges even if they turn out to be useless, (0,0) 
// ASSUMPTION: get data values first, then phantom ones, in the order detailed below.

  if (!(rangeIport->get(DRangeX))) return;
cout << "FOrce: DRangeX = " << DRangeX->low << "," << DRangeX->high << endl; 
  if (!(rangeIport->get(DRangeY))) return;
cout << "FOrce: DRangeY = " << DRangeY->low << "," << DRangeY->high << endl; 
  if (!(rangeIport->get(DRangeZ))) return;
cout << "FOrce: DRangeZ = " << DRangeZ->low << "," << DRangeZ->high << endl; 
  if (!(rangeIport->get(PRangeX))) return;
cout << "FOrce: PRangeX = " << PRangeX->low << "," << PRangeX->high << endl; 
  if (!(rangeIport->get(PRangeY))) return;
cout << "FOrce: PRangeY = " << PRangeX->low << "," << PRangeX->high << endl; 
  if (!(rangeIport->get(PRangeZ))) return;
cout << "FOrce: PRangeZ = " << PRangeX->low << "," << PRangeX->high << endl; 

// INV: now I have set all the interval values. 


 VoidStarHandle pHandle;
  input->get(pHandle);
  if (!pHandle.get_rep()) return;
  if (!(force = pHandle->getPhantomUVW())) return;

  // now have input force to send to phantom 

// get current position and create/move point to it
  force->updateLock.read_lock();
  Point p1 = force->force.point();
  force->updateLock.read_unlock();
//apply force
  setForce(p1.x(), p1.y(), p1.z());



// LOOP FOR REST OF DATA 
while (1) {

  force->sem.down();
  force->updateLock.read_lock();
  p1 = force->force.point();
  force->updateLock.read_unlock();
//apply force
  setForce(p1.x(), p1.y(), p1.z());
 
}
 
 
} 
//---------------------------------------------------------------











