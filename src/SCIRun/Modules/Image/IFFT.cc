//static char *id="@(#) $Id"

/*
 *  IFFT.cc:  IFFT Module
 *
 *  Written by:
 *    Scott Morris
 *    July 1998
 */

#include <Containers/Array1.h>
#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/GeometryPort.h>
#include <CommonDatatypes/ScalarFieldPort.h>
#include <CoreDatatypes/ScalarFieldRG.h>
#include <CoreDatatypes/ScalarFieldRGfloat.h>
#include <CommonDatatypes/ColorMapPort.h>
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
#include "fftn.h"

namespace SCIRun {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Multitask;

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
   IFFT(const IFFT&, int deep);
   virtual ~IFFT();
   virtual Module* clone(int deep);
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_IFFT(int proc);

};

extern "C" {
  Module* make_IFFT(const clString& id)
    {
      return scinew IFFT(id);
    }
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

IFFT::IFFT(const IFFT& copy, int deep)
: Module(copy, deep)
{
   NOT_FINISHED("IFFT::IFFT");
}

IFFT::~IFFT()
{
}

Module* IFFT::clone(int deep)
{
   return scinew IFFT(*this, deep);
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

static void start_IFFT(void* obj,int proc)
{
  IFFT* img = (IFFT*) obj;

  img->do_IFFT(proc);
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
    int nz = rg->grid.dim3();
    
    newgrid->resize(ny,nx,1);

    np = Task::nprocessors();    

    unsigned long flops,refs;
  
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
// Revision 1.1  1999/07/27 16:58:52  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:32  dav
// Added image files to SCIRun
//
//
