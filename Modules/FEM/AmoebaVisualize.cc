/*
 * AmoebaVisualize: Allows you to visualize the results
 * of the multi-start downhill simplex method applied to
 * the source localization problem.
 *
 * Peter-Pike Sloan
 */

#include <Classlib/NotFinished.h>
#include <Classlib/BitArray1.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/Matrix.h>

#include <Datatypes/KludgeMessage.h>
#include <Datatypes/KludgeMessagePort.h>

#include <Datatypes/ScalarFieldUG.h>

#include <Datatypes/SymSparseRowMatrix.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Multitask/ITC.h>
#include <Multitask/Task.h>
#include <Datatypes/GeometryPort.h>

#include <limits.h>
#include <unistd.h>

#include <Geom/Line.h>
#include <Geom/Sphere.h>
#include <Geom/Group.h>
#include <Geom/Pt.h>
#include <Geom/Material.h>

// just to encapsulate this stuff...

const double CONS_RAD=2.0;
const double SCALE_RAD=10.0;

struct AmoebaGeom {
  Color start;
  Color end; // interpolate linearly between these 2...

  double minV,maxV; // for radius of spheres...

  Array1<GeomSphere*> spheres;
  GeomLines*          lines; // connects all of the spheres...
};

class WakeMeUp;

class AmoebaVisualize : public Module {

  friend class WakeMeUp;

  AmoebaMessageIPort *inupdate;
  
  GeometryOPort       *ogeom;

  Array1<AmoebaGeom>  ambG; // geometry for them...

  AmoebaMessage       ambD; // data for them...

public:
  AmoebaVisualize(const clString& id);
  AmoebaVisualize(const AmoebaVisualize&, int deep);

  virtual ~AmoebaVisualize();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
  Module* make_AmoebaVisualize(const clString& id)
    {
      return scinew AmoebaVisualize(id);
    }
};


AmoebaVisualize::AmoebaVisualize(const clString& id)
: Module("AmoebaVisualize", id, Filter)
{
  // Create the input ports
  inupdate = scinew AmoebaMessageIPort(this,"Amoeba Input", AmoebaMessageIPort::Atomic);
  add_iport(inupdate);

  // create output ports

  ogeom = scinew GeometryOPort(this,"Geometry",GeometryIPort::Atomic);
  add_oport(ogeom);
}

AmoebaVisualize::AmoebaVisualize(const AmoebaVisualize& copy, int deep)
: Module(copy, deep)
{
  NOT_FINISHED("AmoebaVisualize::AmoebaVisualize");
}

AmoebaVisualize::~AmoebaVisualize()
{
}

Module* AmoebaVisualize::clone(int deep)
{
  return scinew AmoebaVisualize(*this, deep);
}

void AmoebaVisualize::execute()
{
  AmoebaMessageHandle amH;

  if (!inupdate->get(amH))
    return;

  if (amH.get_rep() == 0)
    return;

  if (!ambG.size()) { // just initialize everything...
    // only the data - not the values!
    ambG.resize(amH->amoebas.size());
    ambD.amoebas = amH->amoebas;
    ambD.generation = -1;
    double cscale = 1.0/(ambD.amoebas.size()-1);

    if (ambD.amoebas.size() == 1) cscale = 0.7;

    for(int i=0;i<ambD.amoebas.size();i++) {
      
      GeomGroup *ambrs = scinew GeomGroup;
      GeomGroup *ambsrs = scinew GeomGroup;
  
      ambG[i].start = Color(cscale*i,0.2*drand48(),0.2);
      ambG[i].end = Color(cscale*i,0.2*drand48(),0.3);
      ambG[i].spheres.resize(ambD.amoebas[i].sources.size());
      ambG[i].lines = scinew GeomLines();
      ambG[i].lines->pts.resize(ambD.amoebas[i].sources.size()*
				(ambD.amoebas[i].sources.size()-1)
				+ambD.amoebas[i].sources.size()*2);
      
      GeomMaterial *clines = scinew GeomMaterial(ambG[i].lines,
						 (ambG[i].start + ambG[i].end)*0.5);
      ambrs->add(clines);

      ambrs->add(ambsrs);

      ambG[i].minV = 1E+20;
      ambG[i].maxV = -1E+20;

      Color sclr(ambG[i].end.r() - ambG[i].start.r(),
		 ambG[i].end.g() - ambG[i].start.g(),
		 ambG[i].end.b() - ambG[i].start.b());

      for(int j=0;j<ambD.amoebas[i].sources.size();j++) {
	
	ambG[i].spheres[j] = scinew GeomSphere();
	ambG[i].spheres[j]->rad = 10.0; // what the hey...
	ambsrs->add(scinew GeomMaterial(ambG[i].spheres[j],
					ambG[i].start + sclr*(j*cscale)));

	ambD.amoebas[i].generation = -1;
      }
      ogeom->addObj(ambrs,clString("Amoeba ")+to_string(i));
    }
  }

  //cerr << ambD.amoebas.size() << " " << ambD.amoebas[0].sources[0].loc << endl;

  // the data must all be there, so fill it in...
  for(int i=0;i<ambD.amoebas.size();i++) {

    if (amH->amoebas[i].generation > ambD.amoebas[i].generation) {
      ambD.amoebas[i] = amH->amoebas[i];
      int li=0;
      for(int j=0;j<ambD.amoebas[i].sources.size();j++) {
	ambG[i].spheres[j]->cen = ambD.amoebas[i].sources[j].loc;
	Point pt = ambD.amoebas[i].sources[j].loc;

	if (ambD.amoebas[i].sources[j].err < ambG[i].minV)
	  ambG[i].minV = ambD.amoebas[i].sources[j].err;

	if (ambD.amoebas[i].sources[j].err > ambG[i].maxV)
	  ambG[i].maxV = ambD.amoebas[i].sources[j].err;

	for(int k=j+1;k<ambD.amoebas[i].sources.size();k++) {
	  ambG[i].lines->pts[2*li] = pt;
	  ambG[i].lines->pts[2*li +1] = ambD.amoebas[i].sources[k].loc;
	  li++;
	}

	// do the direction as well...

	double sinphi = sin(ambD.amoebas[i].sources[j].phi);

	double theta = ambD.amoebas[i].sources[j].theta;
	double phi = ambD.amoebas[i].sources[j].phi;

	Vector v(cos(theta)*sinphi,sin(theta)*sinphi,cos(phi));

	if (ambD.amoebas[i].sources[j].v < 0) 
	  v = v*(-1); // flip dipole in that case...

	ambG[i].lines->pts[2*li] = pt;
	ambG[i].lines->pts[2*li+1] = pt + v*40;
	li++;
      }
#if 0
      // now fill in the radius of the spheres...

      for(j=0;j<ambD.amoebas[i].sources.size();j++) {
	double val = ambD.amoebas[i].sources[j].err;
	ambG[i].spheres[j]->rad = CONS_RAD + SCALE_RAD*
	  (val-ambG[i].minV)/(ambG[i].maxV-ambG[i].minV);
      }
#endif

    }
  }
  
  ogeom->flushViews();

}
