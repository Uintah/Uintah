//static char *id="@(#) $Id"

/*
 *  Radon.cc:  Radon Projection Module
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
#include <math.h>

using namespace SCICore::Thread;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;

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

  Module* make_Radon(const clString& id)
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
    newgrid=new ScalarFieldRG;
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
    newgrid=new ScalarFieldRG(*rg);

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

    
    newgrid->resize(diag,numtheta,rg->grid.dim3());

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

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.5  1999/09/08 02:27:01  sparker
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
