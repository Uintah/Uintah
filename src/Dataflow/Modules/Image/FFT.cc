//static char *id="@(#) $Id"

/*
 *  FFT.cc:  FFT Module
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
#include "fftn.c"

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;
using namespace SCICore::Multitask;

class FFT : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;

   double min,max;       // Max and min values of the scalarfield


   int np; // number of proccesors
  
public:
   FFT(const clString& id);
   FFT(const FFT&, int deep);
   virtual ~FFT();
   virtual Module* clone(int deep);
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_FFT(int proc);

};

extern "C" {
  Module* make_FFT(const clString& id)
    {
      return scinew FFT(id);
    }
}

static clString module_name("FFT");

FFT::FFT(const clString& id)
: Module("FFT", id, Filter)
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

FFT::FFT(const FFT& copy, int deep)
: Module(copy, deep)
{
   NOT_FINISHED("FFT::FFT");
}

FFT::~FFT()
{
}

Module* FFT::clone(int deep)
{
   return scinew FFT(*this, deep);
}

void FFT::do_FFT(int proc)    
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

static void start_FFT(void* obj,int proc)
{
  FFT* img = (FFT*) obj;

  img->do_FFT(proc);
}

void FFT::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "FFT cannot handle this field type\n";
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
    
    newgrid->resize(ny,nx,2);

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
	if (nz==2)
	  temp2[y*nx+x]=rg->grid(y,x,1); else
	    temp2[y*nx+x]=0;
      }

    int dims[2];
    dims[0] = nx;
    dims[1] = ny;
    
    
    fftnf(2,dims,temp,temp2,1,0.0);
	 
    
      //    fft2d_float(temp, ny, 1, &flops, &refs);
    
    //    Task::multiprocess(np, start_FFT, this);

    for (x=0;x<nx;x++)
      for (int y=0;y<ny;y++) {
	newgrid->grid(y,x,0)=temp[y*nx+x];
	newgrid->grid(y,x,1)=temp2[y*nx+x];
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
// Revision 1.2  1999/08/17 06:39:58  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:51  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:30  dav
// Added image files to SCIRun
//
//

