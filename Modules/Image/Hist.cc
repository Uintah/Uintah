/*
 *  Hist.cc:  Histogram Module
 *
 *  Written by:
 *    Scott Morris
 *    October 1997
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ColorMapPort.h>
#include <Geom/Grid.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <Multitask/Task.h>
#include <math.h>

class Hist : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;

   double min,max;       // Max and min values of the scalarfield


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
};

static clString module_name("Hist");

Hist::Hist(const clString& id)
: Module("Hist", id, Filter)
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
: Module(copy, deep)
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
  int start = (rg->grid.dim2()-1)*proc/np;
  int end   = (proc+1)*(rg->grid.dim2()-1)/np;

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

    rg->get_minmax(min,max);
    
    newgrid->resize(1,max+1,1);

    np = Task::nprocessors();    

  
    cerr << "min/max : " << min << " " << max << "\n";
    
    Task::multiprocess(np, start_hist, this);

    outscalarfield->send( newgrid );
}

/*
void Hist::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/






