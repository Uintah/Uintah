//static char *id="@(#) $Id"

/*
 *  Sharpen.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    Sept 1997
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <iostream>
using std::cerr;


namespace SCIRun {


class Sharpen : public Module
{
  ScalarFieldIPort *inscalarfield0;
  ScalarFieldIPort *inscalarfield1;

  ScalarFieldOPort *outscalarfield;

  int gen0;
  int gen1;

  ScalarFieldRG* newgrid;
  double fact;  // Sharpening factor - "c"


  // Not persistent across execute.
  ScalarFieldRG* rg;
  ScalarFieldRG* blur;
  int np; // number of processors
  
public:

  Sharpen(const clString& id);
  virtual ~Sharpen();
  virtual void execute();
  
  void tcl_command( TCLArgs&, void *);
  
  void do_parallel(int proc);
};


extern "C" Module* make_Sharpen(const clString& id)
{
  return scinew Sharpen(id);
}



Sharpen::Sharpen(const clString& id)
  : Module("Sharpen", id, Filter)
{
  // Create the input ports
  inscalarfield0 =
    scinew ScalarFieldIPort(this, "Scalar Field", ScalarFieldIPort::Atomic);
  inscalarfield1 =
    scinew ScalarFieldIPort(this, "Scalar Field", ScalarFieldIPort::Atomic);

  add_iport( inscalarfield0);
  add_iport( inscalarfield1);

  gen0 = gen1 = 0;

  // Create the output port
  outscalarfield =
    scinew ScalarFieldOPort(this, "Scalar Field", ScalarFieldIPort::Atomic);
  add_oport( outscalarfield);

  fact = 1.0;
}


Sharpen::~Sharpen()
{
  delete inscalarfield0;
  delete inscalarfield1;
  delete outscalarfield;

  // Who's responsible for deleting these?
  if (newgrid) delete newgrid;
  if (rg) delete rg;
  if (blur) delete blur;
}



void
Sharpen::do_parallel(int proc)
{
  int x, y;

  const int start = newgrid->grid.dim1() * proc / np;
  const int end   = (proc + 1) * newgrid->grid.dim1() / np;

  for (x=start; x<end; x++)
  {
    for (y=0; y<newgrid->grid.dim2(); y++)
    {
      newgrid->grid(x, y, 0) =
	fact * rg->grid(x, y, 0) - (1.0 - fact) * blur->grid(x, y, 0);
    }
  }
}

void
Sharpen::execute()
{
  // Get the scalar field.
  ScalarFieldHandle sfield0;
  if (!inscalarfield0->get( sfield0 ))
    return;
    
  rg = sfield0->getRG();
    
  if (!rg)
  {
    cerr << "Sharpen can only handle regular grids.\n";
    return;
  }

  // Get the blurred scalar field.
  ScalarFieldHandle sfield1;
  if (!inscalarfield1->get( sfield1 ))
  {
    cerr << "Sharpen requires a blurred image on the left input port.\n";
    return;
  }  
  blur = sfield1->getRG();
    
  if (!blur)
  {
    cerr << "Sharpen cannot handle this field type\n";
    return;
  }

  // Generation check, if up to date nothing needs to be done.
  // TODO: Gui variables all need to check here also.
  if (gen0 == rg->generation && gen1 == blur->generation)
  {
    // We're all up to date, do nothing.
    return;
  }
  gen0 = rg->generation;
  gen1 = blur->generation;

  const int nx = rg->grid.dim1();
  const int ny = rg->grid.dim2();
  const int nz = rg->grid.dim3();

  if (newgrid) { delete newgrid; }
  newgrid = new ScalarFieldRG(nx, ny, nz);
    
  // TODO:  Maybe this should automatically resample?  Needs cross-module
  if ((nx != blur->grid.dim1()) || (ny != blur->grid.dim2()))
  {
    cerr << "Blurred image must be the same size as input image.\n";
    cerr << "Resample it w/ Subsample module.\n";
    return;
  }
    
  np = Thread::numProcessors();
    
  Thread::parallel(Parallel<Sharpen>(this, &Sharpen::do_parallel), np, true);

  outscalarfield->send( newgrid );
}


void
Sharpen::tcl_command(TCLArgs& args, void* userdata)
{
  // Initialize something.
  if (args[1] == "initmatrix")
  {
    args[2].get_double(fact);
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace SCIRun

