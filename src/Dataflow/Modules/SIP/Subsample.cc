//static char *id="@(#) $Id"

/*
 *  Subsample.cc:  
 *
 *  Written by:
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



#define support (2.0)  // Filter Width

#define B (1.0 / 3.0)  // Mitchell Parameters
#define C (1.0 / 3.0)

typedef struct {
  int pixel;
  double weight;
} CONTRIB;

typedef struct {
  int n;
  CONTRIB *p;
} CLIST;

class Subsample : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid,*tmp;
   ScalarFieldRG*  rg;

   int newx,newy;        // new size of image
   double scalex,scaley;
   int mode;             // Scaling by set size (1) or by factor (2)
   double min,max;       // Max and min values of the scalarfield, used for clamping

   TCLstring funcname;

   CLIST *contrib;

   int np; // number of proccesors
  
public:
   Subsample(const clString& id);
   virtual ~Subsample();
   virtual void execute();

   void tcl_command( TCLArgs&, void *);

   void do_fast(int proc);
   void do_mitchell_row(int proc);
   void do_mitchell_col(int proc);

   void compute_hcontribs();
   void compute_vcontribs();

  
};

extern "C" Module* make_Subsample(const clString& id)
{
   return scinew Subsample(id);
}

//static clString module_name("Subsample");

Subsample::Subsample(const clString& id)
: Module("Subsample", id, Filter),  funcname("funcname",id,this)
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
    scalex=scaley=1.0;  // default to 1 to 1 scale
    mode=2;
}

Subsample::~Subsample()
{
}

double Mitchell(double x)
{
  double xx;

  xx=x*x;
  if (x<0) x=-x;
  if (x<1.0) {
    x = (((12.0 - 9.0 * B - 6.0 * C)*(x*xx)) +
	 ((-18.0 +12.0 *B+6.0 *C)*xx) +
	 (6.0 - 2*B));
    return(x/6.);
  } else if (x<2.0) {
    x = (((-1.0 *B -6.0 *C) *(x*xx)) +
	 ((6.0 *B+30.0 *C)*xx) +
	 ((-12.0 *C-48.0 *C)*x) +
	 (8.0 *B+24 *C));
    return(x/6.);
  }
  return(0.);
}

void Subsample::do_mitchell_row(int proc)
{
  int start = (rg->grid.dim1())*proc/np;
  int end   = (proc+1)*(rg->grid.dim1())/np;

  double weight;
  
  for(int z3=0; z3<newgrid->grid.dim3(); z3++) 
    for(int y=start; y<end; y++) {
      for(int x=0; x<newgrid->grid.dim2(); x++) {
	weight = 0.0;
	for (int z=0;z<contrib[x].n;z++)
	  weight += rg->grid(y,contrib[x].p[z].pixel,z3) *
	    contrib[x].p[z].weight;
	if ((weight>=min) && (weight<=max))     // Clamp to min..max
	  tmp->grid(y,x,z3) = weight; else
	    if (weight<min) tmp->grid(y,x,z3)=min; else
	      if (weight>max) tmp->grid(y,x,z3)=max;
      }
    }
}

void Subsample::do_mitchell_col(int proc)
{
  int start = (newgrid->grid.dim2())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim2())/np;

  double weight;

  for(int z3=0; z3<newgrid->grid.dim3(); z3++) 
    for(int x=start; x<end; x++) {
      for(int y=0; y<newgrid->grid.dim1(); y++) {
	weight = 0.0;
	for (int z=0;z<contrib[y].n;z++)
	  weight += tmp->grid(contrib[y].p[z].pixel,x,z3) *
	    contrib[y].p[z].weight;
	if ((weight>=min) && (weight<=max))
	  newgrid->grid(y,x,z3) = weight; else
	    if (weight<min) newgrid->grid(y,x,z3)=min; else
	      if (weight>max) newgrid->grid(y,x,z3)=max; 
      }
    }
}

void Subsample::do_fast(int proc)    // No filtering, but fast
{
  int start = (newgrid->grid.dim2())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim2())/np;

  int sy,sx;
  
  for(int z=0; z<newgrid->grid.dim3(); z++) 
    for(int x=start; x<end; x++) {
      sx = ((x*rg->grid.dim2())/newx);
      for(int y=0; y<newgrid->grid.dim1(); y++) {
	sy=((y*rg->grid.dim1())/newy);
	newgrid->grid(y,x,z) = rg->grid(sy,sx,z);
      }
    }
}

void Subsample::compute_hcontribs()
{
    double center,right,left;
    double width,fscale,weight;
    int i,j,k,n;
    
    contrib = new CLIST[newx];
    if (scalex < 1.0) {
      width = support / scalex;
      fscale = 1.0 / scalex;
      for (i=0; i<newx; i++) {
	contrib[i].n = 0;
	contrib[i].p = new CONTRIB[(int)width*2+1];
	center = (double) i/scalex;
	left = ceil(center-width);
	right = floor(center+width);
	for (j=left; j<=right; j++) {
	  weight = center - (double) j;
	  weight = (Mitchell(weight/fscale) / fscale);
	  if (j<0) 
	    n = -j; else
	      if (j>=rg->grid.dim2())
		n = (rg->grid.dim2() - j) + rg->grid.dim2()-1; else
		  n = j;
	  k = contrib[i].n++;
	  contrib[i].p[k].pixel = n;
	  contrib[i].p[k].weight = weight;
	}
      }
    } else {
      for (i=0;i<newx;++i) {
	contrib[i].n = 0;
	contrib[i].p = new CONTRIB[(int)support*2+1];
	center = (double) i/scalex;
	left = ceil(center-support);
	right = floor(center+support);
	for (j=left; j<=right; ++j) {
	  weight = center - (double) j;
	  weight = Mitchell(weight);
	  if (j<0) 
	    n = -j; else
	      if (j>=rg->grid.dim2())
		n = (rg->grid.dim2() - j) + rg->grid.dim2()-1; else
		  n = j;
	  k = contrib[i].n++;
	  contrib[i].p[k].pixel = n;
	  contrib[i].p[k].weight = weight;
	}
      }
    }
}

void Subsample::compute_vcontribs()
{
    double center,right,left;
    double width,fscale,weight;
    int i,j,k,n;
    
    contrib = new CLIST[newy];
    if (scaley < 1.0) {
      width = support / scaley;
      fscale = 1.0 / scaley;
      for (i=0; i<newy; i++) {
	contrib[i].n = 0;
	contrib[i].p = new CONTRIB[(int)width*2+1];
	center = (double) i/scaley;
	left = ceil(center-width);
	right = floor(center+width);
	for (j=left; j<=right; j++) {
	  weight = center - (double) j;
	  weight = (Mitchell(weight/fscale) / fscale);
	  if (j<0) 
	    n = -j; else
	      if (j>=rg->grid.dim1())
		n = (rg->grid.dim1() - j) + rg->grid.dim1()-1; else
		  n = j;
	  k = contrib[i].n++;
	  contrib[i].p[k].pixel = n;
	  contrib[i].p[k].weight = weight;
	}
      }
    } else {
      for (i=0;i<newy;++i) {
	contrib[i].n = 0;
	contrib[i].p = new CONTRIB[(int)support*2+1];
	center = (double) i/scaley;
	left = ceil(center-support);
	right = floor(center+support);
	for (j=left; j<=right; ++j) {
	  weight = center - (double) j;
	  weight = Mitchell(weight);
	  if (j<0) 
	    n = -j; else
	      if (j>=rg->grid.dim1())
		n = (rg->grid.dim1() - j) + rg->grid.dim1()-1; else
		  n = j;
	  k = contrib[i].n++;
	  contrib[i].p[k].pixel = n;
	  contrib[i].p[k].weight = weight;
	}
      }
    }
}

void Subsample::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Subsample cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
    //  newgrid=new ScalarFieldRGint;
      // New input
    }
    if (mode==2) {
      newx = rg->grid.dim2()*scalex;
      newy = rg->grid.dim1()*scaley;
    } else {
      scalex = double(newx)/rg->grid.dim2();
      scaley = double(newy)/rg->grid.dim1();
    }
    cerr << "Re-sampling to  : " << newx << " " << newy << "\n";
    cerr << "Scale Factors   : " << scalex << " " << scaley << "\n";

    newgrid = new ScalarFieldRG(newy, newx, rg->grid.dim3());
    
    np = Thread::numProcessors();    

    clString ft(funcname.get());

    rg->compute_minmax();
    rg->get_minmax(min,max);

    cerr << "min/max : " << min << " " << max << "\n";
    
    if (ft=="Mitchell") {

      tmp = new ScalarFieldRG(rg->grid.dim1(), newx, rg->grid.dim3());
    
      compute_hcontribs();  // Horizontal contributions
      Thread::parallel(Parallel<Subsample>(this, &Subsample::do_mitchell_row),
		       np, true);
      
      compute_vcontribs();
      Thread::parallel(Parallel<Subsample>(this, &Subsample::do_mitchell_col),
		       np, true);
    }

    if (ft=="Fast") {
      Thread::parallel(Parallel<Subsample>(this, &Subsample::do_fast),
		       np, true);
    }

    outscalarfield->send( newgrid );
}


void Subsample::tcl_command(TCLArgs& args, void* userdata)
{
  if (args[1] == "initsize") { 
    args[2].get_int(newx);
    args[3].get_int(newy);
    mode=1;
  } else 
    if (args[1] == "initscale") {
      args[2].get_double(scalex);
      args[3].get_double(scaley);
      mode=2;
    } else {
      Module::tcl_command(args, userdata);
    }
}

} // End namespace SCIRun

