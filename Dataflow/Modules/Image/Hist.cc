//static char *id="@(#) $Id"

/*
 *  Hist.cc:  Histogram Module
 *
 *  Written by:
 *    Scott Morris
 *    October 1997
 */

#include <Containers/Array1.h>
#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/GeometryPort.h>
#include <CommonDatatypes/ScalarFieldPort.h>
#include <CoreDatatypes/ScalarFieldRG.h>
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

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;

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
   Hist(const Hist&, int deep);
   virtual ~Hist();
   virtual Module* clone(int deep);
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_hist(int proc);

};

extern "C" {
Module* make_Hist(const clString& id)
{
   return scinew Hist(id);
}
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

Hist::Hist(const Hist& copy, int deep)
: Module(copy, deep), includeval("include",id,this),
  numbinsval("numbins",id,this)
{
   NOT_FINISHED("Hist::Hist");
}

Hist::~Hist()
{
}

Module* Hist::clone(int deep)
{
   return scinew Hist(*this, deep);
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

static void start_hist(void* obj,int proc)
{
  Hist* img = (Hist*) obj;

  img->do_hist(proc);
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

    float bindiv = (max/numbins);

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

