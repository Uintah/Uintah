/*
 *  FFT.cc:  FFT Module
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

class FFT : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;

   double min,max;       // Max and min values of the scalarfield


   int np; // number of proccesors
  
public:
   FFT(const clString& id);
   FFT(const FFT&, int deep);
   virtual ~FFT();
   virtual Module* clone(int deep);
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_FFT(int proc);

};

extern "C" {
Module* make_FFT(const clString& id)
{
   return scinew FFT(id);
}
};

static clString module_name("FFT");

FFT::FFT(const clString& id)
: Module("FFT", id, Filter)
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

FFT::FFT(const FFT& copy, int deep)
: Module(copy, deep)
{
   NOT_FINISHED("FFT::FFT");
}

FFT::~FFT()
{
}

Module* FFT::clone(int deep)
{
   return scinew FFT(*this, deep);
}

void FFT::do_FFT(int proc)    
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

static void start_FFT(void* obj,int proc)
{
  FFT* img = (FFT*) obj;

  img->do_FFT(proc);
}

void FFT::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "FFT cannot handle this field type\n";
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
    
    Task::multiprocess(np, start_FFT, this);

    outscalarfield->send( newgrid );
}

/*
void FFT::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/






