//static char *id="@(#) $Id"

/*
 *  Hist.cc:  Histogram Module
 *
 *  Written by:
 *    Scott Morris
 *    October 1997
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
#include <math.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;

class Hist : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;

   double min,max;       // Max and min values of the scalarfield
   TCLint includeval,numbinsval;

   int include,numbins;

   int np; // number of proccesors
  
public:
   Hist(const clString& id);
   virtual ~Hist();
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_hist(int proc);

};

Module* make_Hist(const clString& id)
{
   return scinew Hist(id);
}

//static clString module_name("Hist");

Hist::Hist(const clString& id)
: Module("Hist", id, Filter), includeval("include",id,this),
  numbinsval("numbins",id,this)
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

Hist::~Hist()
{
}

void Hist::do_hist(int proc)    
{
  int start = (rg->grid.dim2())*proc/np;
  int end   = (proc+1)*(rg->grid.dim2())/np;

  for(int x=start; x<end; x++) {
    for(int y=0; y<rg->grid.dim1(); y++) {
      if (rg->grid(y,x,0)>0)
	newgrid->grid(0,rg->grid(y,x,0),0)++;
    }
  }
}

void Hist::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Hist cannot handle this field type\n";
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
    
    //    np = Task::nprocessors();    
  
    cerr << "Hist min/max : " << min << " " << max << "\n";

    include = includeval.get();
    numbins = numbinsval.get();

    newgrid->resize(1,numbins,1);

    for (int i=0; i<numbins; i++)
      newgrid->grid(0,i,0)=0;

    //    float bindiv = (max/numbins);

    for (int x=0; x<rg->grid.dim1(); x++)
      for (int y=0; y<rg->grid.dim2(); y++)
	if (((rg->grid(x,y,0)>=0) && (include)) ||
	    ((rg->grid(x,y,0)>0) && (!include)))
	  if (rg->grid(x,y,0)==0)
	    newgrid->grid(0,0,0)++;
	  else
	    newgrid->grid(0,ceil(float((rg->grid(x,y,0))/max)*numbins)-1,0)++;
    
    //    Task::multiprocess(np, start_hist, this);

    outscalarfield->send( newgrid );
}

/*
void Hist::tcl_command(TCLArgs& args, void* userdata)
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
// Revision 1.3  1999/08/25 03:48:55  sparker
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
// Revision 1.1  1999/04/29 22:26:31  dav
// Added image files to SCIRun
//
//

