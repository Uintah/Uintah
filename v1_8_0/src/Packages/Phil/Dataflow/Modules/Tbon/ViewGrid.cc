
/* ViewGrid.cc
   Display grid points for regular or curvilinear grids

   Packages/Philip Sutton
   June 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#include <Core/Util/NotFinished.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomSphere.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ColorMapPort.h>


namespace Phil {
using namespace SCIRun;

class ViewGrid : public Module {
public:
  // Functions required by Module inheritance
  ViewGrid(const clString& id);
  virtual ~ViewGrid();
  virtual void execute();

protected:
private:
  GuiInt gridtype;
  GuiInt representation;
  GuiDouble radius;
  GuiString geomfilename;
  GuiInt numzones;
  GuiInt showzone1, showzone2, showzone3, showzone4;
  GuiString nx1, ny1, nz1, nx2, ny2, nz2, nx3, ny3, nz3, nx4, ny4, nz4;

  GeometryOPort* geomout;
  ColorMapIPort* colorin;
}; // class ViewGrid

// More required stuff...
extern "C" Module* make_ViewGrid(const clString& id){
  return new ViewGrid(id);
}

// Constructor
ViewGrid::ViewGrid(const clString& id)
  : Module("ViewGrid", id, Filter), gridtype("gridtype",id,this),
  representation("representation",id,this), radius("radius",id,this),
  geomfilename("geomfilename",id,this), numzones("numzones",id,this),
  showzone1("showzone1",id,this), showzone2("showzone2",id,this), 
  showzone3("showzone3",id,this), showzone4("showzone4",id,this), 
  nx1("nx1",id,this), ny1("ny1",id,this), nz1("nz1",id,this),
  nx2("nx2",id,this), ny2("ny2",id,this), nz2("nz2",id,this),
  nx3("nx3",id,this), ny3("ny3",id,this), nz3("nz3",id,this),
  nx4("nx4",id,this), ny4("ny4",id,this), nz4("nz4",id,this)
{
  gridtype.set(0);
  representation.set(0);
  radius.set(0.0);

  geomout = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( geomout );
  colorin = new ColorMapIPort(this, "Color Map", ColorMapIPort::Atomic);
  add_iport( colorin );
}

// Destructor
ViewGrid::~ViewGrid() {
}

// Execute - main guts
void
ViewGrid::execute() {
  int i, j, k;
  
  ColorMapHandle cmap;
  int have_cmap = colorin->get( cmap );
  if( have_cmap )
    cmap->Scale(0,1);

  geomout->delAll();
  if( gridtype.get() == 0 ) {
    // regular grid
    int xmax = atoi( nx1.get()() );
    int ymax = atoi( ny1.get()() );
    int zmax = atoi( nz1.get()() );

    if( representation.get() == 0 ) {
      // points
      GeomPts* pts = new GeomPts( 3 * xmax * ymax * zmax );
      if( have_cmap ) 
	pts->colors.resize(3*xmax*ymax*zmax);
      int n = 0;
      for( k = 0; k < zmax; k++ ) {
	for( j = 0; j < ymax; j++ ) {
	  for( i = 0; i < xmax; i++ ) {
	    pts->add( Point( i, j, k ) );
	    if( have_cmap ) {
	      MaterialHandle matl = 
		cmap->lookup( (double)i / ((double)xmax - 1.0) );
	      pts->colors[n] = matl->diffuse.r();
	      pts->colors[n+1] = matl->diffuse.g();
	      pts->colors[n+2] = matl->diffuse.b();
	      n+=3;
	    }
	  }
	}
      }
      geomout->addObj(pts,"GridPoints");
 
    } else {
      // spheres
      GeomGroup* group = new GeomGroup();
      for( k = 0; k < zmax; k++ ) {
	for( j = 0; j < ymax; j++ ) {
	  for( i = 0; i < xmax; i++ ) {
	    GeomSphere* s = new GeomSphere( Point( i, j, k ), radius.get() );
	    if( have_cmap ) {
	      MaterialHandle matl = cmap->lookup((double)i/((double)xmax-1.0));
	      GeomMaterial* m = new GeomMaterial( s, matl );
	      group->add( m );
	    } else {
	      group->add( s );
	    }
	  }
	}
      }
      geomout->addObj(group,"GridPoints");
    }
    
  } else {
    // curvilinear grid
    FILE* geom = fopen( geomfilename.get()(), "r" );
    if( !geom ) {
      cerr << "Error: cannot open file " << geomfilename.get()() << endl;
      return;
    }

    int show[] = {0, 0, 0, 0};
    if( showzone4.get() == 1 ) show[3] = 1;
    if( showzone3.get() == 1 ) show[2] = 1; 
    if( showzone2.get() == 1 ) show[1] = 1; 
    if( showzone1.get() == 1 ) show[0] = 1; 

    int nx[4], ny[4], nz[4];
    nx[0] = atoi( nx1.get()() );
    ny[0] = atoi( ny1.get()() );
    nz[0] = atoi( nz1.get()() );
    if( numzones.get() > 1 ) {
      nx[1] = atoi( nx2.get()() );
      ny[1] = atoi( ny2.get()() );
      nz[1] = atoi( nz2.get()() );
    }
    if( numzones.get() > 2 ) {
      nx[2] = atoi( nx3.get()() );
      ny[2] = atoi( ny3.get()() );
      nz[2] = atoi( nz3.get()() );
    } 
    if( numzones.get() > 3 ) {
      nx[3] = atoi( nx4.get()() );
      ny[3] = atoi( ny4.get()() );
      nz[3] = atoi( nz4.get()() );
    }

    float xyz[3];
    if( representation.get() == 0 ) {
      // points
      for( int zone = 0; zone < numzones.get(); zone++ ) {
	if( show[zone] == 1 ) {
	  GeomPts* pts = new GeomPts(1);
	  if( have_cmap )
	    pts->colors.resize( 3*nx[zone]*ny[zone]*nz[zone] );

	  int n = 0;
	  for( k = 0; k < nz[zone]; k++ ) {
	    for( j = 0; j < ny[zone]; j++ ) {
	      for( i = 0; i < nx[zone]; i++ ) {
		if( fread( xyz, sizeof(float), 3, geom ) != 3 ) 
		  cerr << "Warning!  Extraneous fread!" << endl;
		pts->add( Point( xyz[0], xyz[1], xyz[2] ) );
		if( have_cmap ) {
		  MaterialHandle matl = 
		    cmap->lookup( (double)i / ((double)nx[zone] - 1.0) );
		  pts->colors[n] = matl->diffuse.r();
		  pts->colors[n+1] = matl->diffuse.g();
		  pts->colors[n+2] = matl->diffuse.b();
		  n+=3;
		}
	      }
	    }
	  }
	  geomout->addObj(pts,"GridPoints");
	} else {
	  fseek( geom, sizeof(float)*3*nx[zone]*ny[zone]*nz[zone], SEEK_CUR );
	}
      }
    } else {
      // spheres
      for( int zone = 0; zone < numzones.get(); zone++ ) {
	if( show[zone] == 1 ) {
	  GeomGroup* group = new GeomGroup();
	  for( k = 0; k < nz[zone]; k++ ) {
	    for( j = 0; j < ny[zone]; j++ ) {
	      for( i = 0; i < nx[zone]; i++ ) {
		if( fread( xyz, sizeof(float), 3, geom ) != 3 ) 
		  cerr << "Warning!  Extraneous fread!" << endl;
		GeomSphere* s = new GeomSphere( Point(xyz[0],xyz[1],xyz[2]), 
						radius.get() );
		if( have_cmap ) {
		  MaterialHandle matl = 
		    cmap->lookup( (double)i / ((double)nx[zone]-1.0) );
		  GeomMaterial* m = new GeomMaterial( s, matl );
		  group->add( m );
		} else {
		  group->add( s );
		}
	      }
	    }
	  }
	  geomout->addObj(group,"GridPoints");
	} else {
	  fseek( geom, sizeof(float)*3*nx[zone]*ny[zone]*nz[zone], SEEK_CUR );
	}
      }
    }

    fclose(geom);
  }
  
  
} // execute

} // End namespace Phil


