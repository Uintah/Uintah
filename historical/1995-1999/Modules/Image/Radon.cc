/*
 *  Radon.cc:  Radon Projection Module
 *
 *  Written by:
 *    Scott Morris
 *    July 1998
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ColorMapPort.h>
#include <Geom/Grid.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <Multitask/Task.h>
#include <math.h>

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
   Radon(const Radon&, int deep);
   virtual ~Radon();
   virtual Module* clone(int deep);
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_Radon(int proc);

};

extern "C" {
Module* make_Radon(const clString& id)
{
   return scinew Radon(id);
}
};

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
    newgrid=new ScalarFieldRG;
}

Radon::Radon(const Radon& copy, int deep)
  : Module(copy, deep), higval("higval", id, this),
    lowval("lowval",id,this), num("num",id,this)
{
   NOT_FINISHED("Radon::Radon");
}

Radon::~Radon()
{
}

Module* Radon::clone(int deep)
{
   return scinew Radon(*this, deep);
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

static void start_Radon(void* obj,int proc)
{
  Radon* img = (Radon*) obj;

  img->do_Radon(proc);
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
    newgrid=new ScalarFieldRG(*rg);

    rg->compute_minmax();
    rg->get_minmax(min,max);

    width = rg->grid.dim2();
    height = rg->grid.dim1();
    cx = width/2;
    cy = height/2;

    diag = ceil(sqrt(width*width+height*height));  // Compute the max dimension

    // Get the theta parameters from the tcl window
    
    numtheta = num.get();
    thetamin = lowval.get();
    thetamax = higval.get();
    thetastep = (thetamax-thetamin+1)/numtheta;
    cerr << "cx = " << cx << " cy = " << cy << "\n";
    cerr << "thetastep = " << thetastep << "\n";

    
    newgrid->resize(diag,numtheta,rg->grid.dim3());

    // Run the radon code in parallel
    
    np = Task::nprocessors();
    
    Task::multiprocess(np, start_Radon, this);
    
    outscalarfield->send( newgrid );
}

/*
void Radon::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/






