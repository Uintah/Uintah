/*
 *  Binop.cc:  Binary Operations on ScalarFields
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

class Binop : public Module {
  ScalarFieldIPort *inscalarfield;
  ScalarFieldIPort *inscalarfield2;
  ScalarFieldOPort *outscalarfield;
  int gen;
  
  ScalarFieldRG* newgrid;
  ScalarFieldRG *a,*b;
  
  int mode;             // The number of the operation being preformed
  
  TCLstring funcname;
  
  int np; // number of proccesors
  
public:
  Binop(const clString& id);
  Binop(const Binop&, int deep);
  virtual ~Binop();
  virtual Module* clone(int deep);
  virtual void execute();
  
  void tcl_command( TCLArgs&, void *);
  
  void do_op(int proc);
};

extern "C" {
Module* make_Binop(const clString& id)
{
   return scinew Binop(id);
}
}

//static clString module_name("Binop");

Binop::Binop(const clString& id)
: Module("Binop", id, Filter),  funcname("funcname",id,this)
{
    // Create the input ports
    // Need two scalar fields
  
    inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield);

    inscalarfield2 = scinew ScalarFieldIPort( this, "Scalar Field",
				        ScalarFieldIPort::Atomic);
    add_iport( inscalarfield2);
    // Create the output port
    outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_oport( outscalarfield);
    newgrid=new ScalarFieldRG;
    mode=0;
}

Binop::Binop(const Binop& copy, int deep)
: Module(copy, deep),  funcname("funcname",id,this)
{
   NOT_FINISHED("Binop::Binop");
}

Binop::~Binop()
{
}

Module* Binop::clone(int deep)
{
   return scinew Binop(*this, deep);
}

void Binop::do_op(int proc)    // Do the operation.. paralell
{
  int start = (newgrid->grid.dim2()-1)*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim2()-1)/np;

  for(int z=0; z<newgrid->grid.dim3(); z++) 
    for(int x=start; x<end; x++) {
      for(int y=0; y<newgrid->grid.dim1(); y++) {
	switch (mode) {
	case 0:
	  newgrid->grid(y,x,z) = a->grid(y,x,z) + b->grid(y,x,z);
	  break;
	case 1:  
	  newgrid->grid(y,x,z) = a->grid(y,x,z) - b->grid(y,x,z);
	  break;
	case 2:  
	  newgrid->grid(y,x,z) = (int) a->grid(y,x,z) | (int) b->grid(y,x,z);
	  break;
	case 3:
	  newgrid->grid(y,x,z) = (int) a->grid(y,x,z) & (int) b->grid(y,x,z);
	  break;
	case 4:  
	  newgrid->grid(y,x,z) = Max(a->grid(y,x,z),b->grid(y,x,z));
	  break;
	case 5:  
	  newgrid->grid(y,x,z) = Min(a->grid(y,x,z),b->grid(y,x,z));
	  break;
	}
      }
    }
}

static void start_op(void* obj,int proc)
{
  Binop* img = (Binop*) obj;

  img->do_op(proc);
}

void Binop::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    a=sfield->getRG();

    if (!inscalarfield2->get( sfield ))
	return;

    b=sfield->getRG();
    
    if ((!a) || (!b)) {
      cerr << "Binop cannot handle this field type\n";
      return;
    }

    if ((a->grid.dim1()!=b->grid.dim1()) || (a->grid.dim2()!=b->grid.dim2()) ||
	(a->grid.dim3()!=b->grid.dim3())) {
      cerr << "Dimensions of A and B must agree!";
      return;
    }

    newgrid=new ScalarFieldRG;
    
    newgrid->resize(a->grid.dim1(),a->grid.dim2(),a->grid.dim3());

    np = Task::nprocessors();    

    // see which radio button is pressed..
    
    clString ft(funcname.get());

    if (ft=="A+B") mode=0;
    if (ft=="A-B") mode=1;
    if (ft=="AorB") mode=2;
    if (ft=="AandB") mode=3;
    if (ft=="max(A,B)") mode=4;
    if (ft=="min(A,B)") mode=5;

    cout << "Mode: " << mode << "\n";
    
    Task::multiprocess(np, start_op, this);

    outscalarfield->send( newgrid );
}


void Binop::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}







