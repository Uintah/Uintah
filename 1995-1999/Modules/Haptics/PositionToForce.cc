 // WRITTEN OR MODIFIED BY LDURBECK ldurbeck 6/10/98
// LATER rename this to GenHapticEffect.cc
// LATER make this accept vector fields for irregular grids as well.
// Currently it accepts only VectorFieldRG's, regular grids.

/* PositionToForce.cc: reads the current position and calculates a 
 * matching force. Just does the math/data: Position.cc talks to
 * the phantom to get current position and Force.cc talks to the
 * phantom to set current force. This one just talks to those
 * two modules, not to the phantom directly.
 * 
 * 
 *  
 * Written by Lisa Durbeck with help from Dave Weinstein
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
#include <Datatypes/VoidStarPort.h>
#include <Datatypes/VectorFieldRG.h>
#include <Datatypes/VectorFieldPort.h>
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

#include <Modules/Haptics/PhantomData.h>

class PositionToForce : public Module {
 
public:
  inline Vector calculateForce(Vector &p) {
          // given the current position in p, calculate matching force
  
  Vector v; // stores return value for this function
  if (VinFlag == true) {
     // then pvectors points to the vector field.
     Point pt =   p.point();
     if (pvectors->interpolate(pt, v) == 1){
        if (!(v == oldv)  ) // if it has changed, print it out to stdout
 //          cout << "force:" << v << endl;  
        oldv = v;
     } else {
        //cout << "interpolation failed or negative space." << endl;
        v = Vector(0.0, 0.0, 0.0); 
     }

   } // if VinFlag
  else { 
       // no vector field dataset to work with. Can't calculate a force. 
      // debug/prototype using sinx*siny surface
     double t1, t2, t3;
     t1 = p.x()/17.0;   // scale factor such that the desired range, -pi -> pi,
                   // will distribute nicely over -50  -> 50 phantom mm workspace
     t2 = p.z()/17.0;
     t3 = 50.0*t1*t2; // distribute over range, range = 50
     t1 = p.y() - t3;
     t2 = t1/-100.0; // reverse sign and bound to +-1
     // t2 = t2*3.0;   // scale to reasonable force range: +- 3 N/mm
     v = Vector(0.0, t2, 0.0);  // send matching force to phantom
   } // end else
 
  return v;
}
 
  TCLstring tcl_status;
  PositionToForce(const clString& id);
  PositionToForce(const PositionToForce&, int deep);
  virtual ~PositionToForce();
  virtual Module* clone(int deep);
  virtual void execute();
  void setSpaceRanges(); // my fn to set up space/position Intervals & send em out 
  void setValueRanges(); // my fn to set up value/force Intervals & send em out
 
private:
// I/O ports (not actually ports here: shared memory)
  VoidStarIPort * input;  // using dweinste's VoidStar class to get a generic port
  VoidStarOPort * output;
  PhantomXYZ * xyz;  // input position of phantom (in what coordinate system?)      
  PhantomUVW * uvw;  // output forces u = x component, v = y component, w = z component  
// regular old I/O ports
// for data I use only Vector Fields right now because it's obvious what
// to translate them to for the phantom. LATER do surfaces/other reps??
  VectorFieldIPort* Vin;
  VectorFieldRG* pvectors;
  bool VinFlag;  // set to true if I get a vector field passed in
  Vector oldv;  // previous force vector value for comparisons

// for scaling from data space to phantom space I use intervals
// ASSUMPTION: Intervals are initialized to something. No need to explicitly initialize them.
  Interval *DataSpaceRangeX, *DataSpaceRangeY, *DataSpaceRangeZ;
  Interval *PhantomSpaceRangeX, *PhantomSpaceRangeY, *PhantomSpaceRangeZ;
  IntervalOPort * spaceOport;
// for scaling from data value range to phantom force range I use intervals
  Interval *DataValRangeX, *DataValRangeY, *DataValRangeZ;
  Interval *PhantomForceRangeX, *PhantomForceRangeY, *PhantomForceRangeZ;
  IntervalOPort * valueOport;


}; //class


extern "C" {
  
  Module* make_PositionToForce(const clString& id)
  {
    return new PositionToForce(id);
  }
  
};//extern

//--------------------------
void PositionToForce::setValueRanges()
{
  // set DataValueRangeX --> Z and PhantomValueRangeX --> Z and
 // send them out the valueOport.
 // Force module needs them 

    if (VinFlag == true) {
       // go through the vector field to find the max in each dimension
      double xmax, ymax, zmax;
      xmax = ymax = zmax = 0.0;
      double xmin, ymin, zmin;
      xmin = ymin = zmin = 1000000.0;
      Vector vec;
      for (int i=0; i < pvectors->nx; i++)
        for(int j=0; j < pvectors->ny; j++)
          for(int k=0; k < pvectors->nz; k++) {
        // find the maximum value in the vector -- orientation doesn't matter.
            vec =  pvectors->grid(i,j,k);
            if (vec.x() > xmax ) xmax = vec.x();
            if (vec.y() > ymax ) ymax = vec.y();
            if (vec.z() > zmax ) zmax = vec.z();

            if (vec.x() < xmin) xmin = vec.x();
            if (vec.y() < ymin) ymin = vec.y();
            if (vec.z() < zmin) zmin = vec.z();
           }
      printf("Max found in dataset for x: %g\n", xmax);
      printf("Min found in dataset for x: %g\n", xmin);
      DataValRangeX = scinew Interval(xmin, xmax);
      IntervalHandle oH7(DataValRangeX);
      valueOport->send(oH7);
      printf("Max found in dataset for y: %g\n", ymax);
      printf("Min found in dataset for y: %g\n", ymin);
      DataValRangeY = scinew Interval(ymin, ymax);
      IntervalHandle oH8(DataValRangeY);
      valueOport->send(oH8);
      printf("Max found in dataset for z: %g\n", zmax);
      printf("Min found in dataset for z: %g\n", zmin);
      DataValRangeZ = scinew Interval(zmin, zmax);
      IntervalHandle oH9(DataValRangeZ);
      valueOport->send(oH9);
    }
// LATER MAKE PHANTOM RANGES USER-SCALABLE!
    // tolerable ranges for the phantom: maximum force I want to apply is 8.0 N/mm
    PhantomForceRangeX = scinew Interval(-2.0, 2.0); // sideways force.
    IntervalHandle oH10(PhantomForceRangeX);
    valueOport->send(oH10);
    PhantomForceRangeY = scinew Interval(-2.0, 2.0); // upward/downward
    IntervalHandle oH11(PhantomForceRangeY);
    valueOport->send(oH11);
    PhantomForceRangeZ = scinew Interval(-2.0, 2.0); // toward/away
    IntervalHandle oH12(PhantomForceRangeZ);
    valueOport->send(oH12);



}

void PositionToForce::setSpaceRanges()
{
  // set DataSpaceRangeX --> Z and PhantomSpaceRangeX --> Z and
 // send them out the spaceOport.
// Position module needs them. 

 
   Point dataMin, dataMax;  // ASSUMPTION: these are initialized to something
   if (VinFlag == true) { // if got vector field, can compute bounds from it
    pvectors->compute_bounds();  // not necessary for RG but needed for UG
    pvectors->get_bounds(dataMin, dataMax); // gives two points in space
   }

   // construct intervals from the 2 points and send them off
// NOTE: SENDER MUST KNOW IN WHAT ORDER THESE WILL ARRIVE!! LATER SEPARATE PORTS?
   DataSpaceRangeX = scinew Interval(dataMin.x(), dataMax.x());
   IntervalHandle oH(DataSpaceRangeX);
   spaceOport->send(oH);
   DataSpaceRangeY = scinew Interval(dataMin.y(), dataMax.y());
   IntervalHandle oH2(DataSpaceRangeY);
   spaceOport->send(oH2);
   DataSpaceRangeZ = scinew Interval(dataMin.z(), dataMax.z());
   IntervalHandle oH3(DataSpaceRangeZ);
   spaceOport->send(oH3);

   // construct intervals for the phantom from apriori knowledge
 // FOR NOW I HARD-CODE THE PHANTOM COORDINATE SYSTEM. LATER  IF WE
 // END UP USING THIS CODE FOR OTHER DEVICES, I WANT
 // to PASS IT IN AS AN INITIALIZATION MESSAGE


   PhantomSpaceRangeX = scinew Interval(-75.0, 75.0);
   IntervalHandle oH4(PhantomSpaceRangeX);
   spaceOport->send(oH4);
   PhantomSpaceRangeY = scinew Interval(-75.0, 75.0);
   IntervalHandle oH5(PhantomSpaceRangeY);
   spaceOport->send(oH5);
   PhantomSpaceRangeZ = scinew Interval(-75.0, 75.0);
   IntervalHandle oH6(PhantomSpaceRangeZ);
   spaceOport->send(oH6);


}


//---------------------------------------------------------------
PositionToForce::PositionToForce(const clString& id)
  : Module("PositionToForce", id, Filter),
    tcl_status("tcl_status",id,this)

{
// PUT INITIALIZATION STUFF HERE
// the key ports take in positions and output forces. I used VoidStar
// since I wanted my own data type and it was easy to do it with VoidStar.
    input = scinew VoidStarIPort(this, "PhantomXYZ", VoidStarIPort::Atomic);
    add_iport(input);
    output = scinew VoidStarOPort(this, "PhantomUVW", VoidStarIPort::Atomic);
    add_oport(output);

// optional input of VectorFields: if dataflow diagram sends vector
// fields to this module then it uses them to calculateForce().
  Vin=scinew VectorFieldIPort(this, "Input Data", VectorFieldIPort::Atomic);
  add_iport(Vin);

// always need to scale between data range and phantom range
  spaceOport = scinew IntervalOPort(this, "Space Intervals", IntervalIPort::Atomic);
  add_oport(spaceOport);
  valueOport = scinew IntervalOPort(this, "Data Value Intervals", IntervalIPort::Atomic);
  add_oport(valueOport);


}

//----------------------------------------------------------
PositionToForce::PositionToForce(const PositionToForce& copy, int deep)
  : Module(copy, deep),
    tcl_status("tcl_status",id,this)   
{}

//------------------------------------------------------------
PositionToForce::~PositionToForce(){}

//-------------------------------------------------------------
Module* PositionToForce::clone(int deep)
{
  return new PositionToForce(*this, deep);
}

//--------------------------------------------------------------

void PositionToForce::execute()
{

// DO ACTUAL WORK HERE

   tcl_status.set("In PositionToForce::execute(): this loops forever...");

// optional vector field input: get it if it's there.
  VinFlag = false;
  VectorFieldHandle Vhandle;
  if(Vin->get(Vhandle)) {
     if (Vhandle.get_rep()) { // do I need to check the representation?
      pvectors=Vhandle->getRG();
      if (pvectors) { 
        VinFlag = true; 
      }
     }
    }

// Position module handles the shared memory  and first setting of position;
// this module handles the shared memory and first setting of force. s

    setSpaceRanges();  // share interval info with the other modules which need it
    setValueRanges();
   
// calculate first force based on first position. 
    VoidStarHandle rmHandle;
    input->get(rmHandle);
    if (!rmHandle.get_rep()) return; // check for null pointer 
    if (!(xyz = dynamic_cast<PhantomXYZ*>(rmHandle.get_rep()))) return; // check for contents and
           //assign to xyz
    
    xyz->updateLock.read_lock(); 
     Vector p = xyz->position;
    xyz->updateLock.read_unlock();
    
    uvw = new PhantomUVW;
    uvw->force = calculateForce(p); // okay not to lock because guaranteed first time
    VoidStarHandle oHandle(uvw);
    output->send(oHandle);  // send out first computed force based on first received position (a good/real value)
   

// then sit around in loop 
// updating force whenever xyz changes

  while(1) {
     xyz->sem.down();  // mutual exclusion
     xyz->updateLock.read_lock(); 
     Vector p = xyz->position;
     xyz->updateLock.read_unlock();

     Vector v = calculateForce(p);
     uvw->updateLock.write_lock();  // for atomic write
     uvw->force = v;
     uvw->updateLock.write_unlock(); 
     uvw->sem.up();  // notify others of new force
  }

   // does something need current force too?  Some color display? If so, output that too.



 
   cout << "Done!"<<endl; 
 
} 
//---------------------------------------------------------------











