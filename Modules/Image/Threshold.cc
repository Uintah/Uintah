/*
 *  Threshold.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    August 1997
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Geom/Grid.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <Multitask/Task.h>

class Threshold : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;
   TCLdouble high,low,lowval,medval,higval;
   ScalarFieldRG* newgrid;
   ScalarFieldRG* rg;
   ScalarField* sfield;
   double thresh(double val, double orig);

   int np;

   double h,l,lv,mv,hv;  // Threshold values & cutoffs
  
public:
   Threshold(const clString& id);
   Threshold(const Threshold&, int deep);
   virtual ~Threshold();
   virtual Module* clone(int deep);
   virtual void execute();

   void do_parallel(int proc);
  
};

extern "C" {
Module* make_Threshold(const clString& id)
{
   return scinew Threshold(id);
}
};

static clString module_name("Threshold");
static clString widget_name("Threshold Widget");

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

Threshold::Threshold(const Threshold& copy, int deep)
: Module(copy, deep), 
  low("low", id, this), high("high", id, this), lowval("lowval", id, this),
  medval("medval", id, this), higval("higval", id, this)
{
   NOT_FINISHED("Threshold::Threshold");
}

Threshold::~Threshold()
{
}

Module* Threshold::clone(int deep)
{
   return scinew Threshold(*this, deep);
}

void Threshold::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1()-1)*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1()-1)/np;

  if (!start) start++;
  if (end == newgrid->grid.dim1()-1) end--;

  int z=0;
  
  for(int x=start; x<end; x++)                 
    for(int y=1; y<newgrid->grid.dim2()-1; y++) {
      if (rg->grid(x,y,z) < l) 
	newgrid->grid(x,y,z)=thresh(lv,rg->grid(x,y,z));
      if ((rg->grid(x,y,z) < h) \
	  && (rg->grid(x,y,z) > l))
	    newgrid->grid(x,y,z)=thresh(mv,rg->grid(x,y,z));
      if (rg->grid(x,y,z) > h)
	    newgrid->grid(x,y,z)=thresh(hv,rg->grid(x,y,z));
    }
}

static void do_parallel_stuff(void* obj,int proc)
{
  Threshold* img = (Threshold*) obj;

  img->do_parallel(proc);
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
    
    np = Task::nprocessors();
    Task::multiprocess(np, do_parallel_stuff, this);

    outscalarfield->send( newgrid );
}




