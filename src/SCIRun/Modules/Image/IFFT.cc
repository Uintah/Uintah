//static char *id="@(#) $Id"

/*
 *  IFFT.cc:  IFFT Module
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
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
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
#include "fftn.h"

using namespace SCICore::Thread;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;

class IFFT : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;

   double min,max;       // Max and min values of the scalarfield


   int np; // number of proccesors
  
public:
   IFFT(const clString& id);
   virtual ~IFFT();
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_IFFT(int proc);

};

  Module* make_IFFT(const clString& id)
    {
      return scinew IFFT(id);
    }

static clString module_name("IFFT");

IFFT::IFFT(const clString& id)
: Module("IFFT", id, Filter)
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

IFFT::~IFFT()
{
}

void IFFT::do_IFFT(int proc)    
{
  int start = (rg->grid.dim2()-1)*proc/np;
  int end   = (proc+1)*(rg->grid.dim2()-1)/np;

  for(int x=start; x<end; x++) {
    for(int y=0; y<rg->grid.dim1(); y++) {
      if (rg->grid(y,x,0)>0)
	newgrid->grid(0,rg->grid(y,x,0),0)++;
    }
  }
}

void IFFT::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "IFFT cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
    //  newgrid=new ScalarFieldRGint;
      // New input
    }
    newgrid=new ScalarFieldRG;

    rg->compute_minmax();
    rg->get_minmax(min,max);

    int nx = rg->grid.dim2();
    int ny = rg->grid.dim1();
    //int nz = rg->grid.dim3();
    
    newgrid->resize(ny,nx,1);

    np = Thread::numProcessors();    

    //unsigned long flops,refs;
  
    cerr << "min/max : " << min << " " << max << "\n";

    float *temp = new float[nx*ny];
    float *temp2 = new float[nx*ny];
    
    int x;
    for (x=0;x<nx;x++)
      for (int y=0;y<ny;y++) {
	//	newgrid->grid(x,y,0)=rg->grid(x,y,0);
	temp[y*nx+x]=rg->grid(y,x,0);
	temp2[y*nx+x]=rg->grid(y,x,1);
      }

    int dims[2];
    dims[0] = nx;
    dims[1] = ny;
    
    
    fftnf(2,dims,temp,temp2,-1,(double) nx * (double) ny);
	 
    
      //    fft2d_float(temp, ny, 1, &flops, &refs);
    
    //    Task::multiprocess(np, start_FFT, this);

    for (x=0;x<nx;x++)
      for (int y=0;y<ny;y++) {
	newgrid->grid(y,x,0)=temp[y*nx+x];
	//newgrid->grid(x,y,1)=temp2[y*nx+x];
      }
    
    outscalarfield->send( newgrid );
}

/*
void FFT::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.5  1999/09/08 02:26:59  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:32  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:56  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:59  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:52  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:32  dav
// Added image files to SCIRun
//
//
