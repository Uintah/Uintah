//static char *id="@(#) $Id"

/*
 *  Sharpen.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    Sept 1997
 */

#include <SCICore/Containers/Array1.h>
#include <SCICore/Util/NotFinished.h>
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

class Sharpen : public Module {
  ScalarFieldIPort *inscalarfield;
  ScalarFieldIPort *inscalarfield2;
  ScalarFieldOPort *outscalarfield;
  int gen;
  
  ScalarFieldRG* newgrid;
  ScalarFieldRG*  rg;
  ScalarFieldRG*  blur;
  
  double fact;  // Sharpening factor - "c"
  
  int np; // number of proccesors
  
public:
  Sharpen(const clString& id);
  virtual ~Sharpen();
  virtual void execute();
  
  void tcl_command( TCLArgs&, void *);
  
  void do_parallel(int proc);
};

Module* make_Sharpen(const clString& id)
{
   return scinew Sharpen(id);
}

//static clString module_name("Sharpen");

Sharpen::Sharpen(const clString& id)
: Module("Sharpen", id, Filter)
{
  // Create the input ports
  // Need a scalar field
  
  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					     ScalarFieldIPort::Atomic);
  inscalarfield2 = scinew ScalarFieldIPort( this, "Scalar Field",
					     ScalarFieldIPort::Atomic);
  add_iport( inscalarfield);
  add_iport( inscalarfield2);
  // Create the output port
  outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					    ScalarFieldIPort::Atomic);
  add_oport( outscalarfield);
  newgrid=new ScalarFieldRG;
  
  fact = 1.0;
}

Sharpen::~Sharpen()
{
}

void Sharpen::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1())/np;

  for(int x=start; x<end; x++)                
    for(int y=0; y<newgrid->grid.dim2(); y++){
      newgrid->grid(x,y,0) = fact*rg->grid(x,y,0) - \
	(1 - fact)*blur->grid(x,y,0);
    }
}

void Sharpen::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;
    
    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Sharpen cannot handle this field type\n";
      return;
    }
    ScalarFieldHandle sfield2;
    // get the blurred scalar field...if you can

    if (!inscalarfield2->get( sfield2 )) {
      cerr << "Sharpen requires a blurred image on the left input port..\n";
      return;
    }  
    blur=sfield2->getRG();
    
    if(!blur){
      cerr << "Sharpen cannot handle this field type\n";
      return;
    }

    
    
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
      newgrid=new ScalarFieldRG(*rg);
      //New input..";
    }

    int nx=rg->grid.dim1();
    int ny=rg->grid.dim2();
    int nz=rg->grid.dim3();

    newgrid->resize(nx,ny,nz);
    
    if ((nx!=blur->grid.dim1()) || (ny!=blur->grid.dim2())) {
      cerr << "Blurred image must be the same size as input image..\n";
      cerr << "Resample it w/ Subsample module..\n";
      return;
    }
    
    np = Thread::numProcessors();
    
    Thread::parallel(Parallel<Sharpen>(this, &Sharpen::do_parallel),
		     np, true);

    outscalarfield->send( newgrid );
}

void Sharpen::tcl_command(TCLArgs& args, void* userdata)
{
  if (args[1] == "initmatrix") { // initialize something...
    args[2].get_double(fact);
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.4  1999/08/31 08:55:34  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:58  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:02  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:54  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:34  dav
// Added image files to SCIRun
//
//
