//static char *id="@(#) $Id"

/*
 *  Turk.cc:  
 *
 *  Written by:
 */

#include <SCICore/Containers/Array1.h>
#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
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
#include <math.h>

using namespace SCICore::Thread;

extern double atof();
extern double drand48();

namespace SCIRun {
namespace Modules {

float frand(float,float);

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Math;

#define  STRIPES  1
#define  SPOTS    2
  
#define MAX 200
  
class Turk : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG* rg;
   ScalarFieldRG* outgrid;
   double matrix[9]; // matrix for convolution...

   double normal,t1,t2;

   int np; // number of proccesors

   /* screen stuff */

  int xsize;
  int ysize;
  int psize;
  
  /* simulation variables */
  
  int interval;
  int iterations;
  int value_switch;
  int stripe_flag;
  
  float a[MAX][MAX];
  float b[MAX][MAX];
  float c[MAX][MAX];
  float d[MAX][MAX];
  float e[MAX][MAX];
  
  float da[MAX][MAX];
  float db[MAX][MAX];
  float dc[MAX][MAX];
  float dd[MAX][MAX];
  float de[MAX][MAX];
  
  float ai[MAX][MAX];
  
  float p1,p2,p3;
  
  float diff1,diff2;
  
  float arand;
  float a_steady;
  float b_steady;
  
  float beta_init;
  float beta_rand;
  
  float speed;
  
  int sim;
  
public:
   Turk(const clString& id);
   virtual ~Turk();
   virtual void execute();

   void tcl_command( TCLArgs&, void *);

   void do_parallel(int proc);
   void do_stripes();
   void do_spots();
   void semi_equilibria();
   void show(float values[MAX][MAX]);
  
};

  Module* make_Turk(const clString& id)
    {
      return scinew Turk(id);
    }

static clString module_name("Turk");

Turk::Turk(const clString& id)
: Module("Turk", id, Filter)
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
    for (int x=0;x<9;x++)
      matrix[x]=1;
    normal=1;

    xsize = 100;
    ysize = 100;
    psize = 4;
    
    /* simulation variables */
    
    interval = 1;
    iterations = 350; // only needs to be about 350 for stripes..
    value_switch = 1;
    stripe_flag = 0;
    speed = 1.0;
    
    sim = 1;
  

  
}

Turk::~Turk()
{
}

void Turk::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1()-1)*proc/(np-1);
  int end   = (proc+1)*(newgrid->grid.dim1()-1)/(np-1)  + 1;


  //  cerr << proc << " : start " << start << " end " << end << "\n";

  
  /*  if (!start) start++; */

  if (end == newgrid->grid.dim1()) end--;
    
  /* for(int x=start; x<end; x++)                 
    for(int y=1; y<newgrid->grid.dim2()-1; y++){ 
      newgrid->grid(x,y,0) = */

  if (stripe_flag) {
      int i,j;
      int iprev,inext,jprev,jnext;
      float aval,bval,cval,dval,eval;
      float ka,kc,kd;
      float temp1,temp2;
      float dda,ddb;
      float ddd,dde;
      
      /* compute change in each cell */
      
      for (i = 0; i < xsize; i++) {
	
	ka = -p1 - 4 * diff1;
	kc = -p2;
	kd = -p3 - 4 * diff2;
	
	iprev = (i + xsize - 1) % xsize;
	inext = (i + 1) % xsize;
	
	for (j = 0; j < ysize; j++) {
	  
	  jprev = (j + ysize - 1) % ysize;
	  jnext = (j + 1) % ysize;
	  
	  aval = a[i][j];
	  bval = b[i][j];
	  cval = c[i][j];
	  dval = d[i][j];
	  eval = e[i][j];
	  
	  temp1 = 0.01 * aval * aval * eval * ai[i][j];
	  temp2 = 0.01 * bval * bval * dval;
	  
	  dda = a[i][jprev] + a[i][jnext] + a[iprev][j] + a[inext][j];
	  ddb = b[i][jprev] + b[i][jnext] + b[iprev][j] + b[inext][j];
	  ddd = d[i][jprev] + d[i][jnext] + d[iprev][j] + d[inext][j];
	  dde = e[i][jprev] + e[i][jnext] + e[iprev][j] + e[inext][j];
	  
	  da[i][j] = aval * ka + diff1 * dda + temp1 / cval;
	  db[i][j] = bval * ka + diff1 * ddb + temp2 / cval;
	  dc[i][j] = cval * kc + temp1 + temp2;
	  dd[i][j] = dval * kd + diff2 * ddd + p3 * aval;
	  de[i][j] = eval * kd + diff2 * dde + p3 * bval;
	}
      }
      
      /* affect change */
      
      for (i = 0; i < xsize; i++)
	for (j = 0; j < ysize; j++) {
	  a[i][j] += (speed * da[i][j]);
	  b[i][j] += (speed * db[i][j]);
	  c[i][j] += (speed * dc[i][j]);
	  d[i][j] += (speed * dd[i][j]);
	  e[i][j] += (speed * de[i][j]);
	}
  } else {
  
  
	
  int i,j,k;
  int iprev,inext,jprev,jnext;
  float aval,bval;
  float ka;
  float dda,ddb;
  float Diff1,Diff2;

  Diff1 = diff1 / 2.0;
  Diff2 = diff2 / 2.0;
  ka = p1 / 16.0;

  /* compute change in each cell */

  for (k=0; k<iterations; k++) {
    if (k % 1000 == 0)
      cerr << "proc #" << proc << "at " << k << "\n";
    if ((k % interval == 0) && (k!=0) && (proc==13)) {
      
      switch (value_switch) {
      case 1:
	show(a);
	break;
      case 2:
	show(b);
	break;
      case 3:
	show(c);
	break;
      case 4:
	show(d);
	break;
      case 5:
	show(e);
	break;
      default:
	cerr << "bad switch in compute: " << value_switch << "\n";
	break;
      }
    }
  
  for (i = start; i < end; i++) {

    iprev = (i + xsize - 1) % xsize;
    inext = (i + 1) % xsize;

    for (j = 0; j < ysize; j++) {

      jprev = (j + ysize - 1) % ysize;
      jnext = (j + 1) % ysize;

      aval = a[i][j];
      bval = b[i][j];

      dda = a[i][jprev] + a[i][jnext] + a[iprev][j] + a[inext][j] - 4 * aval;
      ddb = b[i][jprev] + b[i][jnext] + b[iprev][j] + b[inext][j] - 4 * bval;

      da[i][j] = ka * (16 - aval * bval) + Diff1 * dda;
      db[i][j] = ka * (aval * bval - bval - c[i][j]) + Diff2 * ddb;
    }
  }

  /* affect change */

  for (i = start; i < end; i++)
    for (j = 0; j < ysize; j++) {
      a[i][j] += (speed * da[i][j]);
      b[i][j] += (speed * db[i][j]);
      if (b[i][j] < 0)
	b[i][j] = 0;
    }
  }
  }
}


/******************************************************************************
Run Meinhardt's stripe-formation system.
******************************************************************************/

void Turk::do_stripes()
{
  p1 = 0.04;
  p2 = 0.06;
  p3 = 0.04;

  diff1 = 0.009;
  diff2 = 0.2;

  arand = 0.02;

  sim = STRIPES;
  value_switch = 1;
}


/******************************************************************************
Run Turing reaction-diffusion system.
******************************************************************************/

void Turk::do_spots()
{
  beta_init = 12;
  beta_rand = 0.1;

  a_steady = 4;
  b_steady = 4;

  diff1 = 0.25;
  diff2 = 0.0625;

  p1 = 0.2;
  p2 = 0.0;
  p3 = 0.0;

  sim = SPOTS;
  value_switch = 2;
}

void Turk::semi_equilibria()
{
  int i,j;
  float ainit,binit;
  float cinit,dinit,einit;

  ainit = binit = cinit = dinit = einit = 0;

  /* figure the values */

  switch (sim) {

    case STRIPES:
      for (i = 0; i < xsize; i++) {

	ainit = p2 / (2 * p1);
	binit = ainit;
	cinit = 0.02 * ainit * ainit * ainit / p2;
	dinit = ainit;
	einit = ainit;
	
	for (j = 0; j < ysize; j++) {
	  a[i][j] = ainit;
	  b[i][j] = binit;
	  c[i][j] = cinit;
	  d[i][j] = dinit;
	  e[i][j] = einit;
	  ai[i][j] = 1 + frand (-0.5 * arand, 0.5 * arand);
	}
      }
      break;

    case SPOTS:
      for (i = 0; i < xsize; i++)
	for (j = 0; j < ysize; j++) {
	  a[i][j] = a_steady;
	  b[i][j] = b_steady;
	  c[i][j] = beta_init + frand (-beta_rand, beta_rand);
	}
      break;

    default:
      cerr << "bad case in semi_equilibria\n";
      break;
  }
}

void Turk::show(float values[MAX][MAX])
{
  int i,j;
  //  float output;
  float min =  1e20;
  float max = -1e20;

  /* find minimum and maximum values */

  for (i = 0; i < xsize; i++)
    for (j = 0; j < ysize; j++) {
      if (values[i][j] < min)
	min = values[i][j];
      if (values[i][j] > max)
	max = values[i][j];
    }

  if (min == max) {
    min = max - 1;
    max = min + 2;
  }

  cerr << "min max diff: " << min << " " << max << " " << max - min << "\n";

  /* display the values */

  for (i = 0; i < xsize; i++) {
    for (j = 0; j < ysize; j++) {
      newgrid->grid(i,j,0) = (values[i][j] - min) / (max - min);
      //output = output * 255.0;
      //printf("%f ",output);
    }
    //printf("\n");
  }
  outgrid = scinew ScalarFieldRG(*newgrid);
  outgrid = newgrid;
  outscalarfield->send_intermediate( outgrid );
}


void Turk::execute()
{
  // get the scalar field...if you can

  /*      ScalarFieldHandle sfield
	  
    if (!inscalarfield->get( sfield ))
    return;

    rg=sfield->getRG();
    
  
    if(!rg){
      cerr << "Turk cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
      newgrid=new ScalarFieldRG(*rg);
      // New input
    }


    
    cerr << "--Turk--\nConvolving with:\n";
    for (int j=0;j<9;j++){
      cerr << matrix[j] << " ";
      if (((j+1) % 3)==0)
	cerr << "\n";
    }
    cerr << "Scaling by: "<< normal << "\n--ENDTurk--\n";

    */

   stripe_flag = 1;
  
    if (stripe_flag)
      do_stripes();
    else
      do_spots();
  
    newgrid->resize(xsize,ysize,1);
    np = Thread::numProcessors();

    cerr << "n: " << np << "\n";
    
    int k;
 
    
    /* calculate semistable equilibria */
    
    semi_equilibria();

    //     Task::multiprocess(np, do_parallel_stuff, this);
    
    /* start things diffusing */
    
         for (k = 0; k < iterations; k++) {
     if (k % interval == 0) {
       cerr << "k: " << k << "\n";
	switch (value_switch) {
	case 1:
	  show(a);
	  break;
	case 2:
	  show(b);
	  break;
	case 3:
	  show(c);
	  break;
	case 4:
	  show(d);
	  break;
	case 5:
	  show(e);
	  break;
	default:
	  cerr << "bad switch in compute: " << value_switch << "\n";
	  break;
	}
      }
      
   
      
      switch (sim) {
      case STRIPES:
	//multiplicative_help();
	//	Task::multiprocess(np, do_parallel_stuff, this);
	do_parallel(1);
	break;
      case SPOTS:
	//turing();
	Thread::parallel(Parallel<Turk>(this, &Turk::do_parallel),
			 np, true);
	//			do_parallel(1);
	break;
      default: break;
      }
    }
    
    


       outscalarfield->send( newgrid );
}


void Turk::tcl_command(TCLArgs& args, void* userdata)
{
  if (args[1] == "initmatrix") { // initialize something...
    for(int j=0;j<9;j++) {
      args[j+2].get_double(matrix[j]);
    }
    args[11].get_double(t1);
    args[12].get_double(t2);
    if (t2)
      normal=(t1/t2); else normal=0;
  } else {
    Module::tcl_command(args, userdata);
  }
}

float frand(float min, float max)
{
  return (min + drand48() * (max - min));
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.4  1999/08/31 08:55:36  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:59  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:03  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:56  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:35  dav
// Added image files to SCIRun
//
//
