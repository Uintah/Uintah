//static char *id="@(#) $Id"

/*
 *  Threshold.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    August 1997
 */

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>
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


namespace SCIRun {



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

extern "C" Module* make_Threshold(const clString& id)
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
    
    newgrid = new ScalarFieldRG(nx, ny, nz);
    
    np = Thread::numProcessors();
    Thread::parallel(Parallel<Threshold>(this, &Threshold::do_parallel),
		     np, true);

    outscalarfield->send( newgrid );
}

} // End namespace SCIRun

