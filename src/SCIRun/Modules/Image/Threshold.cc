//static char *id="@(#) $Id"

/*
 *  Threshold.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    August 1997
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
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

using namespace SCICore::Thread;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;

class Threshold : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;
   TCLdouble high,low,lowval,medval,higval;
   ScalarFieldRG* newgrid;
   ScalarFieldRG* rg;
   ScalarField* sfield;
   double thresh(double val, double orig);
   double min,max;

   int np;

   double h,l,lv,mv,hv;  // Threshold values & cutoffs
  
public:
   Threshold(const clString& id);
   virtual ~Threshold();
   virtual void execute();

   void do_parallel(int proc);
  
};

Module* make_Threshold(const clString& id)
{
   return scinew Threshold(id);
}

//static clString module_name("Threshold");
//static clString widget_name("Threshold Widget");

Threshold::Threshold(const clString& id)
: Module("Threshold", id, Filter),
  low("low", id, this), high("high", id, this), lowval("lowval", id, this),
  medval("medval", id, this), higval("higval", id, this)
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
    gen=99;
}

Threshold::~Threshold()
{
}

void Threshold::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1())/np;

  for (int z=0; z<newgrid->grid.dim3(); z++)
    for(int x=start; x<end; x++)                 
      for(int y=0; y<newgrid->grid.dim2(); y++) {
	if (rg->grid(x,y,z) < l) 
	  newgrid->grid(x,y,z)=thresh(lv,rg->grid(x,y,z));
	if ((rg->grid(x,y,z) <= h) \
	    && (rg->grid(x,y,z) >= l))
	  newgrid->grid(x,y,z)=thresh(mv,rg->grid(x,y,z));
	if (rg->grid(x,y,z) > h)
	  newgrid->grid(x,y,z)=thresh(hv,rg->grid(x,y,z));
      }
}

double Threshold::thresh(double val, double orig)
{
  if (val==-1)
    return orig; else return val;
}

void sameval(double val)
{
  if (val==-1)
    cerr << "Same"; else cerr <<  val;
}

void Threshold::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfieldh;
    if (!inscalarfield->get( sfieldh ))
      return;
    
    sfield=sfieldh.get_rep();
    
    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Threshold cannot handle this field type\n";
      return;
    }
    gen=rg->generation;    
    
    if (gen!=newgrid->generation){
      newgrid=new ScalarFieldRG(*rg);
	// cerr << "New input..";
    }
    
    h=high.get();
    l=low.get();
    lv=lowval.get();
    mv=medval.get();
    hv=higval.get();

    rg->compute_minmax();
    rg->get_minmax(min,max);

    cerr << "Threshold min/max : " << min << " " << max << "\n";
    
    cerr << "--Theshold--\n";
    cerr << "Thresholding with : \n";
    cerr << "<" << l << " : ";
    sameval(lv); cerr << "\n";
    cerr << l << " - " << h << " : ";
    sameval(mv); cerr << "\n";
    cerr << ">" << h << " : ";
    sameval(hv); cerr << "\n";
    cerr << "--ENDThreshold--\n";
    
    int nx=rg->grid.dim1();
    int ny=rg->grid.dim2();
    int nz=rg->grid.dim3();
    
    newgrid->resize(nx,ny,nz);
    
    np = Thread::numProcessors();
    Thread::parallel(Parallel<Threshold>(this, &Threshold::do_parallel),
		     np, true);

    outscalarfield->send( newgrid );
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.5  1999/09/08 02:27:02  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:35  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:59  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:03  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:55  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:35  dav
// Added image files to SCIRun
//
//
