//static char *id="@(#) $Id"

/*
 *  ImageConvolve.cc:  
 *
 *  Written by:
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

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Multitask;

class ImageConvolve : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG* rg;
   double matrix[9]; // matrix for convolution...

   double normal,t1,t2;

   int np; // number of proccesors
  
public:
   ImageConvolve(const clString& id);
   ImageConvolve(const ImageConvolve&, int deep);
   virtual ~ImageConvolve();
   virtual Module* clone(int deep);
   virtual void execute();

   void tcl_command( TCLArgs&, void *);

   void do_parallel(int proc);
  
};

extern "C" {
Module* make_ImageConvolve(const clString& id)
{
   return scinew ImageConvolve(id);
}
}

//static clString module_name("ImageConvolve");

ImageConvolve::ImageConvolve(const clString& id)
: Module("ImageConvolve", id, Filter)
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
}

ImageConvolve::ImageConvolve(const ImageConvolve& copy, int deep)
: Module(copy, deep)
{
   NOT_FINISHED("ImageConvolve::ImageConvolve");
}

ImageConvolve::~ImageConvolve()
{
}

Module* ImageConvolve::clone(int deep)
{
   return scinew ImageConvolve(*this, deep);
}

void ImageConvolve::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1()-1)*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1()-1)/np;

  if (!start) start++;
  if (end == newgrid->grid.dim1()-1) end--;
    
  for(int x=start; x<end; x++)                 // 1 -> newgrid->grid.dim1()-1 
    for(int y=1; y<newgrid->grid.dim2()-1; y++){               // start -> end
      newgrid->grid(x,y,0) = (matrix[0]*rg->grid(x-1,y-1,0) + \
			      matrix[1]*rg->grid(x,y-1,0) + \
			      matrix[2]*rg->grid(x+1,y-1,0) + \
			      matrix[3]*rg->grid(x-1,y,0) + \
			      matrix[4]*rg->grid(x,y,0) + \
			      matrix[5]*rg->grid(x+1,y,0) + \
			      matrix[6]*rg->grid(x-1,y+1,0) + \
			      matrix[7]*rg->grid(x,y+1,0) + \
			      matrix[8]*rg->grid(x+1,y+1,0))*normal; 
      }
}

static void do_parallel_stuff(void* obj,int proc)
{
  ImageConvolve* img = (ImageConvolve*) obj;

  img->do_parallel(proc);
}

void ImageConvolve::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "ImageConvolve cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
      newgrid=new ScalarFieldRG(*rg);
      // New input
    }

    cerr << "--ImageConvolve--\nConvolving with:\n";
    for (int j=0;j<9;j++){
      cerr << matrix[j] << " ";
      if (((j+1) % 3)==0)
	cerr << "\n";
    }
    cerr << "Scaling by: "<< normal << "\n--ENDImageConvolve--\n";


    int nx=rg->grid.dim1();
    int ny=rg->grid.dim2();
    int nz=rg->grid.dim3();
    newgrid->resize(nx,ny,nz);

    np = Task::nprocessors();
    Task::multiprocess(np, do_parallel_stuff, this);

    outscalarfield->send( newgrid );
}


void ImageConvolve::tcl_command(TCLArgs& args, void* userdata)
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

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.3  1999/08/25 03:48:56  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:00  sparker
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
