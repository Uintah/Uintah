//static char *id="@(#) $Id"

/*
 *  ImageSel.cc:  ImageSel Module
 *
 *  Written by:
 *    Scott Morris
 *    July 1998
 */

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Datatypes/ScalarFieldRGfloat.h>
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



class ImageSel : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;
   TCLdouble seltcl;
   int sel,oldsel;

   double min,max;       // Max and min values of the scalarfield

   int np; // number of proccesors
  
public:
   ImageSel(const clString& id);
   virtual ~ImageSel();
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_ImageSel(int proc);

};

  extern "C" Module* make_ImageSel(const clString& id)
    {
      return scinew ImageSel(id);
    }

static clString module_name("ImageSel");

ImageSel::ImageSel(const clString& id)
: Module("ImageSel", id, Filter), seltcl("sel",id,this)
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
    oldsel=-1;
}

ImageSel::~ImageSel()
{
}

void ImageSel::do_ImageSel(int proc)    
{
  int start = (rg->grid.dim2())*proc/np;
  int end   = (proc+1)*(rg->grid.dim2())/np;

  for(int x=start; x<end; x++) 
    for(int y=0; y<rg->grid.dim1(); y++)
      newgrid->grid(y,x,0)=rg->grid(y,x,sel);
}

void ImageSel::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "ImageSel cannot handle this field type\n";
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

    sel = floor(seltcl.get());

    if (sel<0) sel=0;
    if (sel>nz-1) sel=nz-1;
    
    newgrid->resize(ny,nx,1);

    np = Thread::numProcessors();    

    //if (sel!=oldsel) {
      Thread::parallel(Parallel<ImageSel>(this, &ImageSel::do_ImageSel),
		       np, true);
      //  oldsel=sel;
      // }

    outscalarfield->send( newgrid );
}

/*
void ImageSel::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/

} // End namespace SCIRun

