//static char *id="@(#) $Id"

/*
 *  Radon.cc:  Radon Projection Module
 *
 *  Written by:
 *    Scott Morris
 *    July 1998
 */

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Geom/GeomGrid.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <iostream>
using std::cerr;
#include <math.h>


namespace SCIRun {



class Radon : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;
   int diag,width,height;
   

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;

   double min,max;       // Max and min values of the scalarfield
  double thetamin,thetamax,thetastep;   // Radon variables
   int numtheta;
   int cx,cy;

   TCLdouble higval,lowval,num;
   
   int np; // number of proccesors
  
public:
   Radon(const clString& id);
   virtual ~Radon();
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_Radon(int proc);

};

  extern "C" Module* make_Radon(const clString& id)
    {
      return scinew Radon(id);
    }

static clString module_name("Radon");

Radon::Radon(const clString& id)
: Module("Radon", id, Filter), higval("higval", id, this),
  lowval("lowval",id,this), num("num",id,this)
{
    // Create the input ports
    // Need a scalar field
  
    inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield);

    // Create the output port
    outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_oport( outscalarfield);
}

Radon::~Radon()
{
}

void Radon::do_Radon(int proc)    
{
  int start = (numtheta)*proc/np;
  int end = (proc+1)*(numtheta)/np;
  double theta,st,ct,thetar;
  int x,y,sum;

  // The actual Radon code
  // It essentially goes over a set of new points and applies
  // an inverse rotation to get the original points.
  
  for (int ntheta=start; ntheta<end; ntheta++) {
    theta = thetamin+(ntheta*thetastep);
    thetar = (M_PI*theta)/180;
    st = sin(thetar);
    ct = cos(thetar);
    for (int xp=0; xp<diag; xp++) {
      sum = 0;
      for (int yp=0; yp<diag; yp++) {
        x = (ct*(xp-cx) + st*(yp-cy))+cx;
	y = (-st*(xp-cx) + ct*(yp-cy))+cy;
	if ((x<width) && (x>0) && (y<height) && (y>0))
	  sum+=rg->grid(y,x,0);
      }
      newgrid->grid(xp,ntheta,0)=sum;
    }
  }  
}

void Radon::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Radon cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
    //  newgrid=new ScalarFieldRGint;
      // New input
    }

    rg->compute_minmax();
    rg->get_minmax(min,max);

    width = rg->grid.dim2();
    height = rg->grid.dim1();
    cx = width/2;
    cy = height/2;

    diag = ceil(sqrt((double)width*width+height*height));  // Compute the max dimension

    // Get the theta parameters from the tcl window
    
    numtheta = num.get();
    thetamin = lowval.get();
    thetamax = higval.get();
    thetastep = (thetamax-thetamin+1)/numtheta;
    cerr << "cx = " << cx << " cy = " << cy << "\n";
    cerr << "thetastep = " << thetastep << "\n";

    
    newgrid = new ScalarFieldRG(diag, numtheta, rg->grid.dim3());

    // Run the radon code in parallel
    
    np = Thread::numProcessors();
    Thread::parallel(Parallel<Radon>(this, &Radon::do_Radon),
		     np, true);
    
    outscalarfield->send( newgrid );
}

/*
void Radon::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/

} // End namespace SCIRun

