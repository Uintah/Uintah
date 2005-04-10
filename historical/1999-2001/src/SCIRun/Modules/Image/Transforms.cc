//static char *id="@(#) $Id"

/*
 *  Transforms.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    July 1998
 */

#include <SCICore/Containers/Array1.h>
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
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Containers/Array3.h>
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
using namespace SCICore::Math;
using namespace SCICore::Containers;

class Transforms : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;
   TCLdouble coef,perr,spread;
   TCLint inverse;
   TCLstring mode,transmode;
   ScalarFieldRG* newgrid;
   ScalarFieldRG* rg,*sn;
   ScalarField* sfield;
   void haar();
   Array2<double> trans;
   Array2<double> a,b,result,clip;

   int snaxels,numfound;
   int np;
   double clipval;

   double maxx,maxy,resx,resy;  // Transforms parameters
   int fixed,iter;
   int nx,ny,nz;

   int nonzero();
   
public:
   Transforms(const clString& id);
   virtual ~Transforms();
   virtual void execute();
   void do_parallel(int proc);
   void do_clip(int proc);   
};

  extern "C" Module* make_Transforms(const clString& id)
    {
      return scinew Transforms(id);
    }

static clString module_name("Transforms");
static clString widget_name("Transforms Widget");

Transforms::Transforms(const clString& id)
: Module("Transforms", id, Filter),
  coef("coef", id, this), mode("mode", id, this), perr("perr", id, this),
  spread("spread", id, this), inverse("inverse", id, this),
  transmode("trans", id, this)
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
    gen=99;
    snaxels=0;
}

Transforms::~Transforms()
{
}

void Transforms::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1())/np;

  for(int y=start; y<end; y++)                 
    for(int x=0; x<newgrid->grid.dim2(); x++) {
      for (int xx=0; xx<nx; xx++)
	result(y,x)+=a(xx,x)*b(y,xx);
    }
}

void Transforms::do_clip(int proc)
{
  int start = (newgrid->grid.dim1())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1())/np;

  for(int y=start; y<end; y++)                 
    for(int x=0; x<newgrid->grid.dim2(); x++) {
      if (Abs(result(y,x))>clipval)
	clip(y,x)=result(y,x); else
	  clip(y,x)=0;
    }
}

double pow(double x,double y) {
  double res = x;
  for (int i=1; i<y; i++)
    res*=x;
  if (y==0) res = 1;
  return res;
}

void Transforms::haar()
{
  int x,y,z;
  
  for (x=0; x<nx; x++)
    trans(0,x)=1;
  double *coef = new double[nx];
  coef[0]=1;
  for (x=2; x<=nx; x++) 
    coef[x-1]=sqrt(pow(2,x-2));
  double iter = log10((double)nx)/log10((double)2);
  for (x=1; x<=iter; x++) {
    int subiter = pow(2,x-1);
    int len = nx/subiter;
    double *subm = new double[len];
    for (y=0; y<len; y++)
      if (y<len/2)
	subm[y]=coef[x]; else
	  subm[y]=-coef[x];
    for (y=1; y<=subiter; y++)
      for (z=1; z<=len; z++)
	trans((pow(2,x-1)+y)-1,(z+(y-1)*len)-1)=subm[z-1];
  }
  for (x=0;x<nx;x++)
    for (y=0;y<nx;y++)
      trans(x,y)/=sqrt((double)nx);
}

int Transforms::nonzero() {
  int res=0;
  
  for (int x=0; x<nx; x++)
    for (int y=0; y<nx; y++)
      if (clip(x,y)!=0)
	res++;
  return res;
}


void Transforms::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfieldh;
    if (!inscalarfield->get( sfieldh ))
      return;
    
    sfield=sfieldh.get_rep();
    
    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Transforms cannot handle this field type\n";
      return;
    }

    /*   rg->resize(2,2,1);
    rg->grid(0,0,0)=1;
    rg->grid(0,1,0)=2;
    rg->grid(1,0,0)=3;
    rg->grid(1,1,0)=4; */
    
    gen=rg->generation;    
    
    if (gen!=newgrid->generation){
      newgrid=new ScalarFieldRG(*rg);
    }
    
    double coeft = coef.get();
    //double perrt = perr.get();
    double spreadt = spread.get();
    int inverset = inverse.get();
    
    cerr << "--Transforms--\n";
    cerr << "\n"; 

    nx=rg->grid.dim1();
    ny=rg->grid.dim2();
    nz=rg->grid.dim3();

    if (((log((double)nx)/log(2.))-floor(log((double)nx)/log(2.))) ||
	((log((double)nx)/log(2.))-floor(log((double)nx)/log(2.)))) { 
      cerr << "Image to be transformed must powers of two in width&height..\n";
      return;
    }

    newgrid->resize(nx,ny,nz);

    /*    np = Task::nprocessors();
    Task::multiprocess(np, do_parallel_stuff, this); */

    trans.newsize(nx,nx);

    clString transt(transmode.get());
    clString modet(mode.get());
    
    haar();

    Array2<double> trans2;

    trans2.newsize(nx,nx);
    int x;
    for (x=0; x<nx; x++)
	for (int y=0; y<nx; y++)
	  trans2(y,x)=trans(y,x);

    // transpose the matrix if we're doing the inverse transform
    
    if (inverset) {
      cerr << "Preforming inverse transform..\n";
      for (x=0; x<nx; x++)
	for (int y=0; y<nx; y++)
	  trans(y,x)=trans2(x,y);
    }
    
    /*for (int i=0;i<nx;i++) {
      for (int j=0;j<nx;j++)
	cerr << trans(i,j) << " ";
      cerr << "\n";
      }  */ 
     
    //Do the matrix multiplication
    
    np = Thread::numProcessors();

    a = trans;
    b.newsize(nx,nx);
    result.newsize(nx,nx);

    for (x=0; x<nx; x++)
      for (int y=0; y<nx; y++) {
	  b(y,x)=rg->grid(y,x,0);
	  result(y,x)=0;
      }
    
    Thread::parallel(Parallel<Transforms>(this, &Transforms::do_parallel),
		     np, true);
    
    for (x=0; x<nx; x++)
      for (int y=0; y<nx; y++) {
	  a(y,x)=result(y,x);
	  result(y,x)=0;
      }
    b = trans;
    Thread::parallel(Parallel<Transforms>(this, &Transforms::do_parallel),
		     np, true);

    if (!inverset) {
    
      double first,last,mid;
      
      first=last=Abs(result(0,0));
      clip.newsize(nx,nx);
      for (x=0; x<nx; x++)
	for (int y=0; y<nx; y++) {
	  last = Max(last,Abs(result(x,y)));
	  first = Min(first,Abs(result(x,y)));
	  clip(y,x) = result(y,x);
	}
      
      int nz = nonzero();
      int found = (nz<coeft);
      int oldnz = nz;
      
      while ((!found) && (first < last) && (Abs(coeft-nz) >spreadt)) {
	mid = (first+last)/2;
	
	clipval = mid;  cerr << "clipval : " << clipval << "\n";
	Thread::parallel(Parallel<Transforms>(this, &Transforms::do_clip),
			 np, true);
	
	nz = nonzero();
	if (nz==oldnz) break;
	oldnz=nz;

	if (coeft>nz)
	  last = mid; else first = mid;
      }
      
      for (x=0; x<nx; x++)
	for (int y=0; y<nx; y++)
	  newgrid->grid(y,x,0)=clip(y,x);
    } else
      for (x=0; x<nx; x++)
	for (int y=0; y<nx; y++)
	  newgrid->grid(y,x,0)=result(y,x);
      
    cerr << "Done w/ Transform!\n";
    
    outscalarfield->send( newgrid );
      
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.7  2000/03/17 09:29:07  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:08:17  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/08 02:27:03  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:35  sparker
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
