/*
 * Lets you view the bounding box and boundaries of a scalar field
 * Peter-Pike Sloan
 */


#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ColorMapPort.h>
#include <Geom/QMesh.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <TCL/TCLvar.h>

#include <iostream.h>

class BoundGrid : public Module {
  ScalarFieldIPort *inscalarfield;
  ColorMapIPort *incolormap;
  GeometryOPort* ogeom;

  TCLint use_lines;  // wether to use lines or not...

  int grid_id; // id of the grid
  
  ScalarField* sfield;
  ScalarFieldRG* ingrid;
  ColorMap* cmap;

  GeomQMesh* grid[6]; // ptrs to all of the grids - so you can change alpha
public:
  BoundGrid(const clString& id);
  BoundGrid(const BoundGrid&, int deep);
  virtual ~BoundGrid();
  virtual Module* clone(int deep);
  virtual void execute();

//  virtual void tcl_command(TCLArgs&, void*);
};
extern "C" {
Module* make_BoundGrid(const clString& id)
{
   return scinew BoundGrid(id);
}
}
//static clString module_name("BoundGrid");

BoundGrid::BoundGrid(const clString& id)
: Module("BoundGrid", id, Filter), 
  use_lines("use_lines",id,this)
{
  // Create the input ports
  // Need a scalar field and a colormap
  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
                                          ScalarFieldIPort::Atomic);
  add_iport( inscalarfield);
  incolormap = scinew ColorMapIPort( this, "ColorMap",
                                    ColorMapIPort::Atomic);
  add_iport( incolormap);
  
  // Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", 
                               GeometryIPort::Atomic);
  add_oport(ogeom);

  grid_id=0;

  for(int i=0;i<6;grid[i++] = 0)
    ; 
  // grids are initialized now...
}

BoundGrid::BoundGrid(const BoundGrid& copy, int deep)
: Module(copy, deep),   use_lines("use_lines",id,this)
{
  NOT_FINISHED("BoundGrid::BoundGrid");
}

BoundGrid::~BoundGrid()
{
}

Module* BoundGrid::clone(int deep)
{
  return scinew BoundGrid(*this, deep);
}

void BoundGrid::execute()
{
  int old_grid_id = grid_id;
  
  cerr << "Starting\n";

  // get the scalar field and colormap...if you can
  ScalarFieldHandle sfieldh;
  if (!inscalarfield->get( sfieldh ))
    return;
  sfield=sfieldh.get_rep();
  
  if (!sfield->getRG())
    return;
  
  ingrid = sfield->getRG();
  
  ColorMapHandle cmaph;
  use_lines.reset(); // what is going on...
  int do_lines=1; // cludge for now..
  if (!use_lines.get()) {
    do_lines=0;
    if (!incolormap->get( cmaph ))
      return;
    cmap=cmaph.get_rep();
  } // only matters if you aren't using lines
  
  cerr << do_lines << " Got here!\n";

  if (do_lines) { // just extract the boundarys...
    // make line segments for this guy...

    GeomLines *nlines = scinew GeomLines;

    int nx = ingrid->nx;
    int ny = ingrid->ny;
    int nz = ingrid->nz;
    
    if (ingrid->is_augmented) {
      // loop through all of the edges...
      for(int j=0;j<ny-1;j++) {
	Point pazt(ingrid->get_point(0,j,nz-1)),pbzt(ingrid->get_point(0,j+1,nz-1));
	Point pazb(ingrid->get_point(0,j,0)),pbzb(ingrid->get_point(0,j+1,0));

	Point paxt(ingrid->get_point(nx-1,j,nz-1)),pbxt(ingrid->get_point(nx-1,j+1,nz-1));
	Point paxb(ingrid->get_point(nx-1,j,0)),pbxb(ingrid->get_point(nx-1,j+1,0));

	nlines->add(pazt,pbzt);
	nlines->add(pazb,pbzb);
	nlines->add(paxt,pbxt);
	nlines->add(paxb,pbxb);
      }

      for(j=0;j<nx-1;j++) {
	Point pazt(ingrid->get_point(j,nx-1,nz-1)),pbzt(ingrid->get_point(j+1,nx-1,nz-1));
	Point pazb(ingrid->get_point(j,0,nz-1)),pbzb(ingrid->get_point(j+1,0,nz-1));

	Point paxt(ingrid->get_point(j,ny-1,0)),pbxt(ingrid->get_point(j+1,ny-1,0));
	Point paxb(ingrid->get_point(j,0,0)),pbxb(ingrid->get_point(j+1,0,0));

	nlines->add(pazt,pbzt);
	nlines->add(pazb,pbzb);
	nlines->add(paxt,pbxt);
	nlines->add(paxb,pbxb);
      }

      for(j=0;j<nz-1;j++) {
	Point pazt(ingrid->get_point(0,0,j)),pbzt(ingrid->get_point(0,0,j+1));
	Point pazb(ingrid->get_point(0,ny-1,j)),pbzb(ingrid->get_point(0,ny-1,j+1));

	Point paxt(ingrid->get_point(nx-1,0,j)),pbxt(ingrid->get_point(nx-1,0,j+1));
	Point paxb(ingrid->get_point(nx-1,ny-1,j)),pbxb(ingrid->get_point(nx-1,ny-1,j+1));

	nlines->add(pazt,pbzt);
	nlines->add(pazb,pbzb);
	nlines->add(paxt,pbxt);
	nlines->add(paxb,pbxb);
      }
    } else {
      nlines->add(ingrid->get_point(0,0,0),ingrid->get_point(nx-1,0,0));
      nlines->add(ingrid->get_point(0,0,0),ingrid->get_point(0,ny-1,0));
      nlines->add(ingrid->get_point(0,0,0),ingrid->get_point(0,0,nz-1));
      
      nlines->add(ingrid->get_point(nx-1,ny-1,nz-1),ingrid->get_point(nx-1,0,nz-1));
      nlines->add(ingrid->get_point(nx-1,ny-1,nz-1),ingrid->get_point(0,ny-1,nz-1));
      nlines->add(ingrid->get_point(nx-1,ny-1,nz-1),ingrid->get_point(nx-1,ny-1,0));

      nlines->add(ingrid->get_point(nx-1,0,0),ingrid->get_point(nx-1,ny-1,0));
      nlines->add(ingrid->get_point(nx-1,0,0),ingrid->get_point(nx-1,0,nz-1));

      nlines->add(ingrid->get_point(nx-1,0,nz-1),ingrid->get_point(0,0,nz-1));

      nlines->add(ingrid->get_point(0,ny-1,0),ingrid->get_point(0,ny-1,nz-1));
      nlines->add(ingrid->get_point(0,ny-1,0),ingrid->get_point(nx-1,ny-1,0));

      nlines->add(ingrid->get_point(0,ny-1,nz-1),ingrid->get_point(0,0,nz-1));
    }
    // delete the old grid/cutting plane
    if (old_grid_id != 0)
      ogeom->delObj( old_grid_id );
    
    grid_id = ogeom->addObj(nlines, "Bounding Box");

  } else { // do the "color" planes...
    // allocate the 6 grids...

    grid[0] = scinew GeomQMesh(ingrid->nx,ingrid->ny);
    grid[1] = scinew GeomQMesh(ingrid->nx,ingrid->ny);

    grid[2] = scinew GeomQMesh(ingrid->nx,ingrid->nz);
    grid[3] = scinew GeomQMesh(ingrid->nx,ingrid->nz);

    grid[4] = scinew GeomQMesh(ingrid->ny,ingrid->nz);
    grid[5] = scinew GeomQMesh(ingrid->ny,ingrid->nz);

    cerr << "Allocated grids!\n";

    // now you just have to run through all of the points...
    // shade with the gradients...

    double cdenom = 1.0/(cmap->max-cmap->min); //scale it

    for(int j = 0; j < ingrid->ny; j++) {
      for(int i=0;i < ingrid->nx; i++) {
	Point pb = ingrid->get_point(i,j,0);
	Vector vb = ingrid->gradient(i,j,0);

	Point pt = ingrid->get_point(i,j,ingrid->nz-1);
	Vector vt = ingrid->gradient(i,j,ingrid->nz-1);

	//Color cb;

	double sb,st;

	sb = (ingrid->grid(i,j,0)-cmap->min)*cdenom;
	st = (ingrid->grid(i,j,ingrid->nz-1)-cmap->min)*cdenom;

	MaterialHandle hb = cmap->lookup2(sb);
	MaterialHandle ht = cmap->lookup2(st);

	grid[0]->add(i,j,pb,vb,hb->diffuse);
	grid[1]->add(i,j,pt,vt,ht->diffuse);
      }
    } // do the xy grids...


    for(j = 0; j < ingrid->nz; j++) {
      for(int i=0;i < ingrid->nx; i++) {
	Point pb = ingrid->get_point(i,0,j);
	Vector vb = ingrid->gradient(i,0,j);

	Point pt = ingrid->get_point(i,ingrid->ny-1,j);
	Vector vt = ingrid->gradient(i,ingrid->ny-1,j);

	//Color cb;

	double sb,st;

	sb = (ingrid->grid(i,0,j)-cmap->min)*cdenom;
	st = (ingrid->grid(i,ingrid->ny-1,j)-cmap->min)*cdenom;

	MaterialHandle hb = cmap->lookup2(sb);
	MaterialHandle ht = cmap->lookup2(st);

	grid[2]->add(i,j,pb,vb,hb->diffuse);
	grid[3]->add(i,j,pt,vt,ht->diffuse);
      }
    } // do the xy grids...


    for(j = 0; j < ingrid->nz; j++) {
      for(int i=0;i < ingrid->ny; i++) {
	Point pb = ingrid->get_point(0,i,j);
	Vector vb = ingrid->gradient(0,i,j);

	Point pt = ingrid->get_point(ingrid->nx-1,i,j);
	Vector vt = ingrid->gradient(ingrid->nx-1,i,j);

	//Color cb;

	double sb,st;

	sb = (ingrid->grid(0,i,j)-cmap->min)*cdenom;
	st = (ingrid->grid(ingrid->nx-1,i,j)-cmap->min)*cdenom;

	MaterialHandle hb = cmap->lookup2(sb);
	MaterialHandle ht = cmap->lookup2(st);

	grid[4]->add(i,j,pb,vb,hb->diffuse);
	grid[5]->add(i,j,pt,vt,ht->diffuse);
      }
    } // do the xy grids...
    
    // now stick these in a group and you are done...

    cerr << "Forking over...\n";

    GeomGroup *grp = scinew GeomGroup;

    for(int q=0;q<6;q++)
      grp->add(grid[q]);
    
    // delete the old grid/cutting plane
    if (old_grid_id != 0)
      ogeom->delObj( old_grid_id );
    
    grid_id = ogeom->addObj(grp, "Bounding Box");
  }
}


