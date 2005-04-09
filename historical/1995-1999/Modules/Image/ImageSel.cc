/*
 *  ImageSel.cc:  ImageSel Module
 *
 *  Written by:
 *    Scott Morris
 *    July 1998
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldRGfloat.h>
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

class ImageSel : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;
   TCLdouble seltcl;
   int sel,oldsel;

   double min,max;       // Max and min values of the scalarfield

   int np; // number of proccesors
  
public:
   ImageSel(const clString& id);
   ImageSel(const ImageSel&, int deep);
   virtual ~ImageSel();
   virtual Module* clone(int deep);
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_ImageSel(int proc);

};

extern "C" {
Module* make_ImageSel(const clString& id)
{
   return scinew ImageSel(id);
}
};

static clString module_name("ImageSel");

ImageSel::ImageSel(const clString& id)
: Module("ImageSel", id, Filter), seltcl("sel",id,this)
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
    oldsel=-1;
}

ImageSel::ImageSel(const ImageSel& copy, int deep)
: Module(copy, deep), seltcl("sel",id,this)
{
   NOT_FINISHED("ImageSel::ImageSel");
}

ImageSel::~ImageSel()
{
}

Module* ImageSel::clone(int deep)
{
   return scinew ImageSel(*this, deep);
}

void ImageSel::do_ImageSel(int proc)    
{
  int start = (rg->grid.dim2())*proc/np;
  int end   = (proc+1)*(rg->grid.dim2())/np;

  for(int x=start; x<end; x++) 
    for(int y=0; y<rg->grid.dim1(); y++)
      newgrid->grid(y,x,0)=rg->grid(y,x,sel);
}

static void start_ImageSel(void* obj,int proc)
{
  ImageSel* img = (ImageSel*) obj;

  img->do_ImageSel(proc);
}

void ImageSel::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "ImageSel cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
    //  newgrid=new ScalarFieldRGint;
      // New input
    }
    newgrid=new ScalarFieldRG;

    rg->compute_minmax();
    rg->get_minmax(min,max);

    int nx = rg->grid.dim2();
    int ny = rg->grid.dim1();
    int nz = rg->grid.dim3();

    sel = floor(seltcl.get());

    if (sel<0) sel=0;
    if (sel>nz-1) sel=nz-1;
    
    newgrid->resize(ny,nx,1);

    np = Task::nprocessors();    

    //if (sel!=oldsel) {
      Task::multiprocess(np, start_ImageSel, this);
      //  oldsel=sel;
      // }

    outscalarfield->send( newgrid );
}

/*
void ImageSel::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/






