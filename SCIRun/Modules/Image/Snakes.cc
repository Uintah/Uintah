//static char *id="@(#) $Id"

/*
 *  Snakes.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    June 1998
 */

#include <Containers/Array1.h>
#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Geom/GeomGrid.h>
#include <Geom/GeomGroup.h>
#include <Geom/GeomLine.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLvar.h>
#include <Multitask/Task.h>
#include <math.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Multitask;

class ipoint {
public:
  ipoint() { samp=0.0; line = 0.0; }
  int samp,line;
};

class Snakes : public Module {
   ScalarFieldIPort *inscalarfield,*inscalarfield2;
   ScalarFieldOPort *outscalarfield;
   int gen;
   TCLdouble aval,bval,maxxval,maxyval,resxval,resyval,iterval;
   TCLint fixedval;
   ScalarFieldRG* newgrid;
   ScalarFieldRG* rg,*sn;
   ScalarField* sfield;
   int snaxels,numfound;
   double thresh(double val, double orig);
   ipoint *inpoints,*outpoints;
   int np;

   double a,b,maxx,maxy,resx,resy;  // Snakes parameters
   int fixed,iter;
  
public:
   Snakes(const clString& id);
   Snakes(const Snakes&, int deep);
   virtual ~Snakes();
   virtual Module* clone(int deep);
   virtual void execute();

   void do_parallel(int proc);
   void do_find_snaxels(int proc);
   double dosnake(int n, int nmline, int nmsamp, ipoint* points,
	     ipoint* output_points, int max_delta_x, int max_delta_y,
	     int resol_x,
	     int resol_y, float alpha, float beta, int fix_end_points);
  
};

extern "C" {
  Module* make_Snakes(const clString& id)
    {
      return scinew Snakes(id);
    }
}

static clString module_name("Snakes");
static clString widget_name("Snakes Widget");

Snakes::Snakes(const clString& id)
: Module("Snakes", id, Filter),
  aval("a", id, this), bval("b", id, this), maxxval("maxx", id, this),
  maxyval("maxy", id, this), resxval("resx", id, this),
  resyval("resy", id, this), fixedval("fixed", id, this),
  iterval("iter",id,this)
{
    // Create the input ports
    // Need a scalar field
  
    inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield);

    inscalarfield2 = scinew ScalarFieldIPort( this, "Scalar Field2",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield2);
    // Create the output port
    outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_oport( outscalarfield);

    newgrid=new ScalarFieldRG;
    gen=99;
    snaxels=0;
}

Snakes::Snakes(const Snakes& copy, int deep)
: Module(copy, deep),
  aval("a", id, this), bval("b", id, this), maxxval("maxx", id, this),
  maxyval("maxy", id, this), resxval("resx", id, this),
  resyval("resy", id, this), fixedval("fixed", id, this),
  iterval("iter",id,this)
{
   NOT_FINISHED("Snakes::Snakes");
}

Snakes::~Snakes()
{
}

Module* Snakes::clone(int deep)
{
   return scinew Snakes(*this, deep);
}

void Snakes::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1())/np;

  int z=0;
  
  for(int x=start; x<end; x++)                 
    for(int y=0; y<newgrid->grid.dim2(); y++) {
      /*      if (rg->grid(x,y,z) < l) 
	newgrid->grid(x,y,z)=thresh(lv,rg->grid(x,y,z));
      if ((rg->grid(x,y,z) < h) \
	  && (rg->grid(x,y,z) > l))
	    newgrid->grid(x,y,z)=thresh(mv,rg->grid(x,y,z));
      if (rg->grid(x,y,z) > h)
	    newgrid->grid(x,y,z)=thresh(hv,rg->grid(x,y,z)); */
      newgrid->grid(x,y,z)=0; // rg->grid(x,y,z);
      //      if (sn->grid(x,y,z))
      //	snaxels++;
    }
}

static void do_parallel_stuff(void* obj,int proc)
{
  Snakes* img = (Snakes*) obj;

  img->do_parallel(proc);
}

/*
 *The snake() function
 *
 */
double Snakes::dosnake(int n, int nmline, int nmsamp, ipoint* points,
	     ipoint* output_points, int max_delta_x, int max_delta_y,
	     int resol_x,
	     int resol_y, float alpha, float beta, int fix_end_points)
{
    int num_scan_x;
    int num_scan_y;
    int num_states;
    double *Smat;
    int *Imat;
    int *scan_x;
    int *scan_y;
    int *delta_x;
    int *delta_y;
    int *states_x;
    int *states_y;
    int *states_i;
    
    int i,j,ind,k,v1,v2;
    int temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;
    double minSoFar;
    int minIndexSoFar;
    int final_i;
    double e,temp_mat;
    
    /*A sanity check:*/
    if (resol_x < 1)
	resol_x = 1;
    if (resol_y < 1)
	resol_y = 1;

    /*some init needed for memory allocation (and later)*/
    num_scan_x = max_delta_x*2/resol_x + 1;
    num_scan_y = max_delta_y*2/resol_y + 1;
    num_states  = num_scan_y * num_scan_x;

    /*Allocate our needed memory*/
    Smat = (double *)malloc(sizeof(double)*n*num_states*num_states);
    Imat = (int *)malloc(sizeof(int) * n*num_states*num_states);
    scan_x = (int *)malloc(sizeof(int) * num_scan_x);
    scan_y = (int *)malloc(sizeof(int) * num_scan_y);
    delta_x = (int *)malloc(sizeof(int) * num_states);
    delta_y = (int *)malloc(sizeof(int) * num_states);
    states_x = (int *)malloc(sizeof(int) * n*num_states);
    states_y = (int *)malloc(sizeof(int) * n*num_states);
    states_i = (int *)malloc(sizeof(int) * n*num_states);
    if (!Smat || !Imat || !scan_x || !scan_y || !delta_x || !delta_y || !states_x
	|| !states_y || !states_i)
    {
      cerr << "Malloc failed in snake.c\n";
	exit(-1);
    }
    
    /*Setup of the data structures to the correct values*/
    for (i = 0; i < num_scan_x; i++)
	scan_x[i] = -max_delta_x + i*resol_x;
    
    for (i = 0; i < num_scan_y; i++)
	scan_y[i] = -max_delta_y + i*resol_y;	
    
    for (i = 0; i < num_scan_y; i++)
	for (j = 0; j < num_scan_x; j++)
	{
	    ind = j+i*num_scan_x;
	    delta_x[ind] = scan_x[i];
	    delta_y[ind] = scan_y[j];
	}
    
    if (fix_end_points) /*fix the end points by handling i=0 and i=n-1 special*/
    {
	temp1 = 1;
	temp2 = n-1;
	/*handle two end points special by making states_x & states_y
	 *just be the points instead of any offset*/
	for (j=0;j<num_states;j++)
	{
	    temp3 = num_states*(n-1)+j;
	    states_x[j] = points[0].samp;
	    states_y[j] = points[0].line;
	    states_x[temp3] = points[n-1].samp;
	    states_y[temp3] = points[n-1].line;
	    states_i[j] = (states_x[j])*nmline + states_y[j];
	    states_i[temp3] = (states_x[temp3])*nmline + states_y[temp3];
	}
    }
    else /*not fixing endpoints, so let loop below go over all points*/
    {
	temp1 = 0;
	temp2 = n;
    }
    
    /*This is still initialization -- here we're initializing states_x,
     *states_y, and states_i
     */
    for (i = temp1; i < temp2; i++)
	for (j = 0; j< num_states; j++)
	{
	    ind = i*num_states+j;
	    states_x[ind] = points[i].samp + delta_x[j];
	    states_y[ind] = points[i].line + delta_y[j];
	    if (states_x[ind] < 0)
		states_x[ind] = 0;
	    else
		if (states_x[ind] >= nmsamp)
		    states_x[ind] = nmsamp - 1;
	    if (states_y[ind] < 0)
		states_y[ind] = 0;
	    else
		if (states_y[ind] >= nmline)
		    states_y[ind] = nmline - 1;
	    states_i[ind] = (states_x[ind])*nmline + states_y[ind];
	}	
    
    /*Initialize Smat and Imat*/
    for (i = 0; i < n*num_states*num_states; i++)
    {
	Smat[i] = 0.0;
	Imat[i] = 0;
    }
    
    /*
     *Now we start our actual process of snaking via dynamic programming
     *
     */
    
    
    /* forward pass */
    for (j = 0; j < num_states; j++)
    {
	/*col*/k = (states_i[j]) / nmline;
	/*row*/v1 = (states_i[j]) - k*nmline;
	for (v2 = 0; v2 < num_states; v2++)
	  // 	    Smat[v2*num_states + j] = - pixel(input_image,v1,k);
	  Smat[v2*num_states + j] = - rg->grid(v1,k,0);
    }
    
    for (k = 1; k < n-1; k++)
    {
	ind = (k-1)*num_states;
	for (v2 = 1; v2 <= num_states; v2++)
	{
	    for (v1 = 1; v1 <= num_states; v1++)
	    {
		i = k*num_states+v1-1;
		minIndexSoFar = (k+1)*num_states+v2-1;
		temp8 = states_x[i];
		temp5 = states_y[i];
		temp6 = states_x[minIndexSoFar] -
		    2*states_x[i];
		temp7 = states_y[minIndexSoFar] -
		    2 * states_y[i];
		minIndexSoFar = 0;
		minSoFar =  999999999999999999.9999;
		j = (k-1)*num_states*num_states+(v1-1)*num_states;
		for (i = 0; i < num_states; i++)
		{
		    temp1 = temp8 -	states_x[ind+i];
		    temp1 *= temp1;
		    temp2 =  temp5 - states_y[ind+i];
		    temp2 *= temp2;
		    temp3 =  temp6 + states_x[ind+i];
		    temp3 *= temp3;
		    temp4 =  temp7 + states_y[ind+i];
		    temp4 *= temp4;
		    temp_mat =
			Smat[j + i] +
			alpha * (double) (temp1 + temp2) +
			beta * (double) (temp3 + temp4);
		    if (temp_mat < minSoFar)
		    {
			minSoFar = temp_mat;
			minIndexSoFar = i;
		    }
		}
		Imat[k*num_states*num_states + (v2-1)*num_states+v1-1] =
		    minIndexSoFar;
		/*col*/i = (states_i[k*num_states+v1-1]) / nmline;
		/*row*/minIndexSoFar = (states_i[k*num_states+v1-1]) - i*nmline;
		Smat[k*num_states*num_states + (v2-1)*num_states+v1-1] =
		    minSoFar - rg->grid(minIndexSoFar,i,0);
	    }
	}
    }
    
    for (v1 = 0; v1 < num_states; v1++)
    {
	minIndexSoFar = 0;
	minSoFar = 999999999999999.99;
	j = (n-1)*num_states + v1;
	temp3 = states_x[j];
	temp4 = states_y[j];
	v2 = (n-2)*num_states;
	ind = (n-2)*num_states*num_states + v1*num_states;
	for (i = 0; i < num_states; i++)
	{
	    temp1 = temp3 - states_x[v2 + i];
	    temp1 *= temp1;
	    temp2 = temp4 - states_y[v2 + i];
	    temp2 *= temp2;
	    temp_mat = Smat[ind + i] +
		alpha * (double) (temp1 + temp2);
	    if (temp_mat < minSoFar)
	    {
		minSoFar = temp_mat;
		minIndexSoFar = i;
	    }
	}
	Imat[(n-1)*num_states*num_states + v1] = minIndexSoFar;
	/*col*/i = (states_i[(n-1)*num_states+v1]) / nmline;
	/*row*/minIndexSoFar = (states_i[(n-1)*num_states+v1]) - i*nmline;
	Smat[(n-1)*num_states*num_states + v1] =
	    minSoFar - rg->grid(minIndexSoFar,i,0);
    }
    
    minSoFar = 999999999999.99;
    minIndexSoFar = 0;
    ind = (n-1)*num_states*num_states;
    for (i = 0; i< num_states; i++)
    {
	if (Smat[ind + i] < minSoFar)
	{
	    minIndexSoFar = i;
	    minSoFar = Smat[ind + i];
	}
    }
    
    e = minSoFar;
    final_i = minIndexSoFar;
    
    /*
     *Now we've found our lowest final state, so we do our backwards
     *pass to get the actual path (contour) that produces this lowest
     *final state
     */
    output_points[n-1].samp = states_x[(n-1)*num_states + final_i];
    output_points[n-1].line = states_y[(n-1)*num_states + final_i];
    v1 = final_i;
    v2 = 0;
    for (k = n-2; k>= 0; k--)
    {
	i = Imat[(k+1)*num_states*num_states + (v2)*num_states + v1];
	v2 = v1;
	v1 = i;
	output_points[k].samp = states_x[k*num_states + v1];
	output_points[k].line = states_y[k*num_states + v1];
    }
    free( Smat );
    free( Imat );
    free( scan_x );
    free( scan_y );
    free( delta_x );
    free( delta_y );
    free( states_x );
    free( states_y );
    free( states_i );
    
    return e;
}    





void Snakes::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfieldh;
    if (!inscalarfield->get( sfieldh ))
      return;
    
    sfield=sfieldh.get_rep();
    
    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Snakes cannot handle this field type\n";
      return;
    }

    if (!inscalarfield2->get( sfieldh ))
      return;
    sfield=sfieldh.get_rep();

    sn=sfield->getRG();

    if(!sn){
      cerr << "We need a snake Image!\n";
      return;
    }
    
    gen=rg->generation;    
    
    if (gen!=newgrid->generation){
      newgrid=new ScalarFieldRG(*rg);
    }
    
    a = aval.get();
    b = bval.get();
    maxx = maxxval.get();
    maxy = maxyval.get();
    resx = resxval.get();
    resy = resyval.get();
    fixed = fixedval.get();
    iter = iterval.get();
    
    cerr << "--Snakes--\n";
    cerr << "Snakify!!\n";

     
    int nx=rg->grid.dim1();
    int ny=rg->grid.dim2();
    int nz=rg->grid.dim3();

    int maxpoints=sn->grid.dim1();
    
    newgrid->resize(nx,ny,nz);

    np = Task::nprocessors();
    Task::multiprocess(np, do_parallel_stuff, this);

    
    snaxels = 0;
    
    int ii;
    for (ii=0; ii<maxpoints; ii++)
      if ((sn->grid(ii,0,0)) && (sn->grid(ii,1,0)))
	snaxels++;
    
    inpoints = new ipoint[snaxels];
    outpoints = new ipoint[snaxels];
    
    for (ii=0; ii<snaxels; ii++) {
      inpoints[ii].line=sn->grid(ii,1,0);
      inpoints[ii].samp=sn->grid(ii,0,0);
    }
        
    /*    for (int i=0; i<100; i++) {
      //      inpoints[i].line = (double(rand()/(32767/nx)));
      // inpoints[i].samp = (double(rand()/(32767/nx)));
      inpoints[i].line = 250-i;
      inpoints[i].samp = 201;
      }   */
    
    /*numfound = 0;
    Task::multiprocess(np, find_snaxels, this); */
    
    for (int it=0; it<iter; it++) {
      cerr << "Iteration #" << it << ".\n";
      dosnake(snaxels,nx,ny,inpoints,outpoints,maxx,maxy,resx,resy,a,b,fixed);
      for (int ted=0; ted<snaxels; ted++)
	inpoints[ted] = outpoints[ted];
    }


      for (int i=0; i<snaxels; i++) {
	/*	cerr << "Point " << i << " (" << outpoints[i].line << "," <<
	  outpoints[i].samp << ").\n";  */
	newgrid->grid(outpoints[i].line,outpoints[i].samp,0) = 255;
      } 

      cerr << "Done w/ snakify!\n";
      
      outscalarfield->send( newgrid );

}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.3  1999/08/25 03:48:58  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:02  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:55  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:34  dav
// Added image files to SCIRun
//
//
