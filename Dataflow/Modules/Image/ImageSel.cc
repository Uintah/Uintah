//static char *id="@(#) $Id"

/*
 *  ImageSel.cc:  ImageSel Module
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
#include <iostream>
using std::cerr;
#include <math.h>

using namespace SCICore::Thread;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;

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

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.7  2000/03/17 09:29:04  sparker
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
// Revision 1.4  1999/08/31 08:55:33  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:56  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:00  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:53  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:33  dav
// Added image files to SCIRun
//
//
