//static char *id="@(#) $Id"

/*
 *  Noise.cc:  Add salt&pepper or gaussian noise to an image
 *
 *  Written by:
 *    Scott Morris
 *    Sept 1997
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <SCICore/Geom/GeomGrid.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <iostream>
using std::cerr;
#include <math.h>

using namespace SCICore::Thread;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;

class Noise : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;
  
   ScalarFieldRG* newgrid;
   ScalarFieldRG* rg;
   TCLstring funcname;
  
   double freq,mag;
   double min,max;

   int np; // number of proccesors
  
public:
   Noise(const clString& id);
   virtual ~Noise();
   virtual void execute();

   void tcl_command( TCLArgs&, void *);

   void do_parallel(int proc);
};

extern "C" Module* make_Noise(const clString& id)
{
   return scinew Noise(id);
}

//static clString module_name("Noise");

Noise::Noise(const clString& id)
: Module("Noise", id, Filter), funcname("funcname",id,this)
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

    freq = 0;
    mag = 1;
    min = 0;
    max = 255;
}

Noise::~Noise()
{
}

void Noise::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1())/np;

  for(int x=start; x<end; x++)                
    for(int y=0; y<newgrid->grid.dim2(); y++){
      if (double(rand())/32767 < freq)
	newgrid->grid(x,y,0)=(double(rand())/32767)*mag; else
	    newgrid->grid(x,y,0)=rg->grid(x,y,0);
    }
}

/* Bill Thompson's Normal Distribution Random number generator stuff */

#define SEED1   10000           /* integer between 1 and 30000 */
#define SEED2   10000           /* integer between 1 and 30000 */
#define SEED3   10000           /* integer between 1 and 30000 */
 
static int      seed1 = SEED1;
static int      seed2 = SEED2;
static int      seed3 = SEED3;

double urandom ( void )
/*
 * This implements the Wichman-Hill algorithm as described in Applied
 * Statistics vol. 31 1982 pp 188-190.
 * This algorithm is machine independent. It has a period >2.78*10^13
 * and performs well on the spectral test.
 */
{
  double result;
  
  seed1 = ( 171 * ( seed1 % 177 ) ) - ( 2 * ( seed1 / 177 ) );
  seed2 = ( 172 * ( seed2 % 176 ) ) - ( 35 * ( seed2 / 176 ) );
  seed3 = ( 170 * ( seed1 % 178 ) ) - ( 63 * ( seed1 / 178 ) );
  
  if ( seed1 < 0 )
    seed1 = seed1 + 30269;
  if ( seed2 < 0 )
    seed2 = seed2 + 30307;
  if ( seed3 < 0 )
    seed3 = seed3 + 30323;
  
  result = (( (double) seed1 ) / 30269.0 )
    + (( (double) seed2 ) / 30307.0 )
    + ( ( (double) seed3 ) / 30323.0 );
  result = result - ( (double) ( (int) result ) ); 
  
  return ( result );
}

double nrandom ( void )
/*
 * The above algorithm is used as input to the Beasley-Springer algorithm
 * as described in Applied Statistics vol. 26 1977 pp 118-121 to obtain
 * random values with a normal distribution. 
 */
{
  double result;
  float   zero = 0.0, split = .42, half = 0.5, one = 1.0;
  double  a0 = 2.50662823884, a1 = -18.61500062529, a2 = 41.39119773534;
  double  a3 = -25.44106049637, b1 = -8.47351093099, b2 = 23.08336743743;
  double  b3 = -21.06224101826, b4 = 3.13082909833, c0 = -2.78718931138;
  double  c1 = -2.29796479134, c2 = 4.85014127135, c3 = 2.32121276858;
  double  d1 = 3.54388924762, d2 = 1.63766781897;
  double  q, r;
  double  udist;
  
  udist = urandom ( );
  
  q = udist - half;
  if ( fabs ( q ) <=  split ) 
    {
      r = q * q;
      result = ( q * ((( a3 * r + a2 ) * r + a1 ) * r + a0 ) ) /
	((((b4 * r + b3) * r + b2) * r + b1) * r + one);
      
      result = q * (((((( a3 * r ) + a2 ) * r ) + a1) * r ) + a0 ) /
	((((((( b4 * r ) + b3 ) * r) + b2 ) * r + b1 ) * r )
	 + one );
    }
  else
    {
      r = udist;
      if ( q > zero )
	r = one - udist;
      if ( r > zero)
	{
	  r = sqrt ( -log ( r ) );
	  result = (((((( c3 * r ) + c2 ) * r ) + c1 ) * r ) +
		    c0 ) / (((( d2 * r ) + d1 ) * r ) + one );
	  if( q < zero)
	    result = - result;
	}
      else
	{
	  result = zero;
	}
    }       
  
  return(result);
}


void Noise::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Noise cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
      newgrid=new ScalarFieldRG(*rg);
      //New input..";
    }

    cerr << "Adding noise to image..,\n";
    
    int nx=rg->grid.dim1();
    int ny=rg->grid.dim2();
    int nz=rg->grid.dim3();

    newgrid->resize(nx,ny,nz);

    clString ft(funcname.get());

    if (ft=="Salt&Pepper") {
    
      np = Thread::numProcessors();
      
      Thread::parallel(Parallel<Noise>(this, &Noise::do_parallel),
		       np, true);
    }

    if (ft=="Gaussian") {

      for (int x=0; x<newgrid->grid.dim2(); x++)
	for (int y=0; y<newgrid->grid.dim1(); y++) {
	  newgrid->grid(y,x,0) = rg->grid(y,x,0) + mag*nrandom(); 
	  if (newgrid->grid(y,x,0) < min) newgrid->grid(y,x,0)=min;
	  if (newgrid->grid(y,x,0) > max) newgrid->grid(y,x,0)=max;
	}
    }
    
    outscalarfield->send( newgrid );
}

void Noise::tcl_command(TCLArgs& args, void* userdata)
{
  if (args[1] == "initmatrix") { // initialize something...
    args[2].get_double(freq);
    args[3].get_double(mag);
    args[4].get_double(min);
    args[5].get_double(max);
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.7  2000/03/17 09:29:05  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:08:15  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/08 02:27:00  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:34  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:57  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:01  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:54  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:33  dav
// Added image files to SCIRun
//
//
