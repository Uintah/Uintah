/*
 *  Unop.cc:  Unary Operations on ScalarFields
 *
 *  Written by:
 *    Scott Morris
 *    November 1997
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

class Unop : public Module {
  ScalarFieldIPort *inscalarfield;
  ScalarFieldOPort *outscalarfield;
  int gen;
  
  ScalarFieldRG* newgrid;
  ScalarFieldRG* rg;
  
  int mode;             // The number of the operation being preformed
  double min,max;       // Max and min values of the scalarfield
  
  TCLstring funcname;
  
  int np; // number of proccesors
  
public:
  Unop(const clString& id);
  Unop(const Unop&, int deep);
  virtual ~Unop();
  virtual Module* clone(int deep);
  virtual void execute();
  
  void tcl_command( TCLArgs&, void *);
  
  void do_op(int proc);
};

extern "C" {
Module* make_Unop(const clString& id)
{
   return scinew Unop(id);
}
};

static clString module_name("Unop");

Unop::Unop(const clString& id)
: Module("Unop", id, Filter),  funcname("funcname",id,this)
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
    mode=0;
}

Unop::Unop(const Unop& copy, int deep)
: Module(copy, deep),  funcname("funcname",id,this)
{
   NOT_FINISHED("Unop::Unop");
}

Unop::~Unop()
{
}

Module* Unop::clone(int deep)
{
   return scinew Unop(*this, deep);
}

void Unop::do_op(int proc)    // Do the operations in paralell
{
  int start = (newgrid->grid.dim2()-1)*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim2()-1)/np;

  for(int z=0; z<newgrid->grid.dim3(); z++) 
    for(int x=start; x<end; x++) {
      for(int y=0; y<newgrid->grid.dim1(); y++) {
	switch (mode) {
	case 0:
	  newgrid->grid(y,x,z) = abs(rg->grid(y,x,z));
	  break;
	case 1:  
	  newgrid->grid(y,x,z) = -rg->grid(y,x,z);
	  break;
	case 2:  
	  newgrid->grid(y,x,z) = (max-min)-rg->grid(y,x,z);
	  break;
	case 3:
	  newgrid->grid(y,x,z) = rg->grid(y,x,z);
	}
      }
    }
}

static void start_op(void* obj,int proc)
{
  Unop* img = (Unop*) obj;

  img->do_op(proc);
}

void Unop::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Unop cannot handle this field type\n";
      return;
    }

    newgrid=new ScalarFieldRG;
    
    newgrid->resize(rg->grid.dim1(),rg->grid.dim2(),rg->grid.dim3());

    np = Task::nprocessors();    

    clString ft(funcname.get());

    if (ft=="Abs") mode=0;
    if (ft=="Negative") mode=1;
    if (ft=="Invert") mode=2;
    if (ft=="Max/Min") mode=3;

    rg->get_minmax(min,max);

    cerr << "min/max : " << min << " " << max << "\n";
    
    Task::multiprocess(np, start_op, this);

    outscalarfield->send( newgrid );
}


void Unop::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}







