/*
 *  Sharpen.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    Sept 1997
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
  Sharpen(const Sharpen&, int deep);
  virtual ~Sharpen();
  virtual Module* clone(int deep);
  virtual void execute();
  
  void tcl_command( TCLArgs&, void *);
  
  void do_parallel(int proc);
};

extern "C" {
Module* make_Sharpen(const clString& id)
{
   return scinew Sharpen(id);
}
};

static clString module_name("Sharpen");

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

Sharpen::Sharpen(const Sharpen& copy, int deep)
: Module(copy, deep)
{
  NOT_FINISHED("Sharpen::Sharpen");
}

Sharpen::~Sharpen()
{
}

Module* Sharpen::clone(int deep)
{
  return scinew Sharpen(*this, deep);
}

void Sharpen::do_parallel(int proc)
{
  int start = (newgrid->grid.dim1()-1)*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim1()-1)/np;

  for(int x=start; x<end; x++)                
    for(int y=0; y<newgrid->grid.dim2(); y++){
      newgrid->grid(x,y,0) = fact*rg->grid(x,y,0) - \
	(1 - fact)*blur->grid(x,y,0);
    }
}

static void do_parallel_stuff(void* obj,int proc)
{
  Sharpen* img = (Sharpen*) obj;

  img->do_parallel(proc);
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
    
    np = Task::nprocessors();
    
    Task::multiprocess(np, do_parallel_stuff, this);

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








