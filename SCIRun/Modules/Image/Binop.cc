//static char *id="@(#) $Id"

/*
 *  Binop.cc:  Binary Operations on ScalarFields
 *
 *  Written by:
 *    Scott Morris
 *    November 1997
 */

#include <SCICore/Containers/Array1.h>
#include <SCICore/Util/NotFinished.h>
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
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <math.h>

using namespace SCICore::Thread;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;

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
  virtual ~Binop();
  virtual void execute();
  
  void tcl_command( TCLArgs&, void *);
  
  void do_op(int proc);
};

Module* make_Binop(const clString& id)
{
   return scinew Binop(id);
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

Binop::~Binop()
{
}

void Binop::do_op(int proc)    // Do the operation.. paralell
{
  int start = (newgrid->grid.dim2())*proc/np;
  int end   = (proc+1)*(newgrid->grid.dim2())/np;

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
	case 6:  
	  newgrid->grid(y,x,z) = a->grid(y,x,z)*b->grid(y,x,z);
	  break;
	case 7:  
	  newgrid->grid(y,x,z) = a->grid(y,x,z)/b->grid(y,x,z);
	  break;
	case 8:  
	  newgrid->grid(y,x,z) = (int) a->grid(y,x,z) ^ (int) b->grid(y,x,z);
	  break;
	}
      }
    }
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

    np = Thread::numProcessors();    

    // see which radio button is pressed..
    
    clString ft(funcname.get());
    
    if (ft=="A+B") mode=0;
    if (ft=="A-B") mode=1;
    if (ft=="AorB") mode=2;
    if (ft=="AandB") mode=3;
    if (ft=="max(A,B)") mode=4;
    if (ft=="min(A,B)") mode=5;
    if (ft=="A*B") mode=6;
    if (ft=="A/B") mode=7;
    if (ft=="AxorB") mode=8;
    

    cout << "Mode: " << mode << "\n";
    
    Thread::parallel(Parallel<Binop>(this, &Binop::do_op),
		     np, true);

    outscalarfield->send( newgrid );
}


void Binop::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.4  1999/08/31 08:55:31  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:54  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:51  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:30  dav
// Added image files to SCIRun
//
//

