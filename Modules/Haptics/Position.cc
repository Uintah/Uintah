 // WRITTEN OR MODIFIED BY LDURBECK ldurbeck 6/11/98

/* Position.cc: first representation of phantom in scirun 
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
#include <iostream.h>
#include <stdio.h>

#ifdef SOCKETS
  #include <Modules/Haptics/client.c>
  //#include <Datatypes/ljdGlobals.h>
#endif

class Position : public Module {
 
public:
  inline int getPosition(double& x, double& y, double& z) {
#ifdef SOCKETS

   double tx, ty, tz;
   read_triple(&tx,&ty,&tz);

// scale to the coordinate system scirun's data knows
 // going from Phantom coord system to Data coord system.
 
       
   if ((PRangeX->low == PRangeX->high) | (PRangeY->low == PRangeY->high)
      | (PRangeZ->low == PRangeZ->high)) {
     cout << "Phantom coordinate system range not set! Incoming position cannot be scaled.\n";
     z = tz;
     y = ty;
     x = tx;   
   } 
 
       
   tx = (tx - PRangeX->low)/(PRangeX->high - PRangeX->low); // where tx lies within PRange
   tx = tx * (DRangeX->high - DRangeX->low); // where x should lie within DRange     
   tx = tx + DRangeX->low;  // starting at the right place

   ty = (ty - PRangeY->low)/(PRangeY->high - PRangeY->low); // where ty lies within PRange
   ty = ty * (DRangeY->high - DRangeY->low); // where y  should lie within DRange
   ty = ty + DRangeY->low;  // starting at the right place

   tz = (tz - PRangeZ->low)/(PRangeZ->high - PRangeZ->low); // where tz lies within PRange
   tz = tz * (DRangeZ->high - DRangeZ->low); // where z should lie within DRange 
   tz = tz + DRangeZ->low;  // starting at the right place
 
   z = tz;
   y = ty;
   x = tx;
   //printf("\t\t(%g, %g, %g)\n", x, y,z);
   return 0;
#else 
// get it from user moving the Endpoint widget around
   return 0;  
#endif 
  

}
 
  TCLstring tcl_status;
  Position(const clString& id);
  Position(const Position&, int deep);
  virtual ~Position();
  virtual Module* clone(int deep);
  virtual void execute();

private:
// I/O ports (not actually ports here: shared memory)
  VoidStarOPort * output;
  PhantomXYZ * position;  // output position of phantom (in what coordinate system?)      
 
  // regular old I/O ports and associated memory
  IntervalHandle DRangeX, DRangeY, DRangeZ; // data space ranges along x,y,and z
  IntervalHandle PRangeX, PRangeY, PRangeZ;  // phantom workspace ranges along x, y, and z  
  IntervalIPort * rangeIport; // to get the intervals from PositionToForce module 
  

}; //class


extern "C" {
  
  Module* make_Position(const clString& id)
  {
    return new Position(id);
  }
  
};//extern


//---------------------------------------------------------------
Position::Position(const clString& id)
  : Module("Position", id, Filter),
    tcl_status("tcl_status",id,this)

{
// PUT INITIALIZATION STUFF HERE
    output = new VoidStarOPort(this, "PhantomXYZ", VoidStarIPort::Atomic);
    add_oport(output);

    rangeIport = scinew IntervalIPort(this, "Force/Value Intervals", IntervalIPort::Atomic);
    add_iport(rangeIport);

}

//----------------------------------------------------------
Position::Position(const Position& copy, int deep)
  : Module(copy, deep),
    tcl_status("tcl_status",id,this)   
{}

//------------------------------------------------------------
Position::~Position(){}

//-------------------------------------------------------------
Module* Position::clone(int deep)
{
  return new Position(*this, deep);
}

//--------------------------------------------------------------

void Position::execute()
{

// DO ACTUAL WORK HERE
//cerr<< "top of Position::execute\n";
//tcl_status.set("Calling Position!");


// get the current force from PositionToForce  
// NOTE: not necessary to couple getting a force to setting a position.
// could have graph with three nodes, one produces phantom positions,
// one translates positions to forces, one
// gets those forces.
// I'm doing it this way for most compact representation, but maybe
// it's misleading this way. Ultimately there will be one phantom process
// which sets position whenever and gets force whenever.

#ifdef SOCKETS
 sock_init();
#endif

double x, y, z;
x = y = z = 0.0;

#ifdef SOCKETS
//ASSERT(getPosition(x,y,z) == 0);
#endif
// start up shared memory
    position = new PhantomXYZ;  // space
    position->position = Vector(x,y,z); // okay not to lock: initialization
    position->sem.up(); 
    position->Esem.up();
    VoidStarHandle pHandle(position);
    output->send(pHandle);   // start up communication

// get range information for converting from phantom to data space

// first I expect a batch of interval/range information so that I
// can convert from data space to phantom space successfully in getPosition().
// ASSUMPTION: I'll get the ranges even if they turn out to be useless, (0,0)
// ASSUMPTION: get data values first, then phantom ones, in the order detailed below.

  if (!(rangeIport->get(DRangeX))) return;
cout << "Position.cc: DRangeX = " << DRangeX->low << "," << DRangeX->high << endl; 
  if (!(rangeIport->get(DRangeY))) return;
cout << "Position.cc: DRangeY = " << DRangeY->low << "," << DRangeY->high << endl; 
  if (!(rangeIport->get(DRangeZ))) return;
cout << "Position.cc: DRangeZ = " << DRangeZ->low << "," << DRangeZ->high << endl; 
  if (!(rangeIport->get(PRangeX))) return;
cout << "Position.cc: PRangeX = " << PRangeX->low << "," << PRangeX->high << endl; 
  if (!(rangeIport->get(PRangeY))) return;
cout << "Position.cc: PRangeY = " << PRangeX->low << "," << PRangeX->high << endl; 
  if (!(rangeIport->get(PRangeZ))) return;
cout << "Position.cc: PRangeZ = " << PRangeX->low << "," << PRangeX->high << endl; 
 
// INV: now I have set all the interval values.


    
// sit in while loop waiting for the last position to be used up and a new one needed
    while ( 1) {
      #ifdef SOCKETS   // if mouse, then Endpoint is getting/setting Position.
       ASSERT(getPosition(x,y,z) == 0);  // read current position
       position->updateLock.write_lock();
       position->position = Vector(x,y,z);
       position->updateLock.write_unlock();
     #endif  // but still notify PositionToForce about it...
       position->sem.try_down(); // nonblocking decrement so that sem doesn't get too big. Also prevents stale data from being used.
       //LATER see what the difference is between uncoupled graphics/force (as it is here) and coupled (by xor'ing the 2 try_downs together) if xor(1,2) then don't update position: wait until both have same value. NOTE: need a new lock around trying the 2 semaphors so that comparing them is atomic.

       position->Esem.try_down();
       position->sem.up(); position->Esem.up();

}

#ifdef SOCKETS
      sock_deinit();
#endif
 
   cout << "Done!"<<endl; 
 
} 


//---------------------------------------------------------------











