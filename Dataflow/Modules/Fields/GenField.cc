/*
 *  GenField.cc:  Unfinished modules
 *
 *  Written by:
 *   Eric Kuehne
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 *
 *  Module GenField
 *
 *  Description:  Generates a Scalar field of doubles according to the
 *  x, y, and z dimensions specified by the user and an arbitrary tcl
 *  equation (function of $x, $y, and $z).  The field's min bound is
 *  initially set to (0,0,0).  
 *
 *  Input ports: None
 *  Output ports: ScalarField 
 */
#include <stdio.h>

#include <SCICore/Datatypes/Field.h>
#include <SCICore/Datatypes/GenSField.h>
#include <SCICore/Datatypes/GenVField.h>
#include <SCICore/Datatypes/LatticeGeom.h>
#include <SCICore/Datatypes/IrregLatticeGeom.h> // Test
#include <SCICore/Datatypes/MeshGeom.h>
#include <SCICore/Datatypes/MeshGeom.h>
#include <SCICore/Datatypes/FlatAttrib.h>
#include <SCICore/Datatypes/AccelAttrib.h>
#include <SCICore/Datatypes/BrickAttrib.h>
#include <SCICore/Datatypes/IndexAttrib.h>
#include <SCICore/Datatypes/AnalytAttrib.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Datatypes/FieldPort.h>
#include <SCICore/Util/DebugStream.h>

#include <PSECore/Dataflow/Module.h>


namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Util;

class GenField : public Module {
private:
  FieldOPort* ofield;
  DebugStream dbg;

  template <class T> void fill(DiscreteAttrib<T> *attrib,
			       LatticeGeom *geom,
			       AnalytAttrib<T> *analyt);

  template <class I, class A> void indicize(I *iatt, A *datt,
					    double minval, double maxval,
					    int x, int y, int z);

  void send_tensor_field();
  void send_vector_field();
  void send_scalar_field();

public:
  //      DebugStream dbg;
      
  TCLint nx;      // determines the size of the field
  TCLint ny;
  TCLint nz;

  TCLdouble tcl_sx, tcl_sy, tcl_sz;  // Start location of field.
  TCLdouble tcl_ex, tcl_ey, tcl_ez;  // End location of field.

  TCLstring fval1; // the equation to evaluate at each point in the field
  TCLstring fval2; // the equation to evaluate at each point in the field
  TCLstring fval3; // the equation to evaluate at each point in the field

  TCLint geomtype;
  TCLint attribtype;
  TCLint indexed;
  
  // constructor, destructor:
  GenField(const clString& id);
  virtual ~GenField();
  FieldHandle fldHandle;
  virtual void execute();
};

extern "C" Module* make_GenField(const clString& id) {
  return new GenField(id);
}

GenField::GenField(const clString& id) :
  Module("GenField", id, Filter), 
  nx("nx", id, this), ny("ny", id, this), nz("nz", id, this),
  tcl_sx("sx", id, this), tcl_sy("sy", id, this), tcl_sz("sz", id, this),
  tcl_ex("ex", id, this), tcl_ey("ey", id, this), tcl_ez("ez", id, this),
  fval1("fval1", id, this),
  fval2("fval2", id, this),
  fval3("fval3", id, this),
  geomtype("geomtype", id, this),
  attribtype("attribtype", id, this),
  indexed("indexed", id, this),
  dbg("GenField", true), fldHandle()
{
  ofield=new FieldOPort(this, "Field", FieldIPort::Atomic);
  add_oport(ofield);
  geomtype.set(1);
  attribtype.set(1);
  indexed.set(0);
}

GenField::~GenField()
{
}


template <class T>
void
GenField::fill(DiscreteAttrib<T> *attrib, LatticeGeom *geom,
	       AnalytAttrib<T> *analyt)
{
  const int x = geom->getSizeX();
  const int y = geom->getSizeY();
  const int z = geom->getSizeZ();

  for (int k=0; k < z; k++)
    {
      for (int j=0; j < y; j++)
	{
	  for (int i=0; i < x; i++)
	    {
	      Point p(i, j, k);
	      Point r;
	      geom->transform(p, r);	
	      attrib->set3(i, j, k, analyt->eval(r.x(), r.y(), r.z()));
	    }
	}
    }
}



template <class I, class A>
void
GenField::indicize(I *iatt, A *datt,
		   double minval, double maxval,
		   int x, int y, int z)
{
  // find min, max, range.  divide up into 256 chunks (transform)
  const double iscale = 255.0 / (maxval - minval);

  for (int i=0; i < 256; i++)
    {
      iatt->tset(i, minval + i / iscale);
    }

  for (int k=0; k < z; k++)
    {
      for (int j=0; j < y; j++)
	{
	  for (int i=0; i < x; i++)
	    {
	      const int val =
		(int)((datt->get3(i, j, k) - minval) * iscale + 0.5);
	      iatt->iset3(i, j, k, val);
	    }
	}
    }
}


#if 0
static void
collect_points(LatticeGeom *geom)
{
  const int x = geom->getSizeX();
  const int y = geom->getSizeY();
  const int z = geom->getSizeZ();

  vector<NodeSimp> nodes;

  for (int k=0; k < z; k++)
    {
      for (int j=0; j < y; j++)
	{
	  for (int i=0; i < x; i++)
	    {
	      Point p(i, j, k);
	      NodeSimp r;
	      geom->transform(p, r.p);	
	      nodes.push_back(r);
	    }
	}
    }
}  


static void
collect_edges(LatticeGeom *geom)
{
  vector<EdgeSimp> edges;

  for (int k=0; k < z; k++)
    {
      for (int j=0; j < y; j++)
	{
	  for (int i=0; i < x; i++)
	    {
	      // TODO: Unmap these from 3d->1d
	      // TODO: compute average values also.
	      if (i+1 < x) { i, j, k  i+1, j, k; }
	      if (j+1 < y) { i, j, k, i, j+1, k; }
	      if (k+1 < z) { i, j, k, i, j, k+1; }
	    }
	}
    }
}    



static void
collect_faces(LatticeGeom *geom)
{
  for (int k=0; k < z; k++)
    {
      for (int j=0; j < y; j++)
	{
	  for (int i=0; i < x; i++)
	    {
	      x 0 y+1, x+1, y+
 		
	    }
	}
    }
}
#endif


void
GenField::send_scalar_field()
{
  // get scalar, vector, tensor:
  const int nfuncts = 1;

  // Get functions.
  vector<string> functs;
  for (int i = 0; i < nfuncts; i++)
    {
      string s = fval1.get()();
      functs.push_back(s);
    }

  // Create analytic attrib.
  // TODO: error checking here?
  AnalytAttrib<double> *analyt = scinew AnalytAttrib<double>();
  analyt->set_fasteval();
  analyt->set_function(functs);

  // Create geometry.
  const int x = nx.get();
  const int y = ny.get();
  const int z = nz.get();

  const Point start(tcl_sx.get(), tcl_sy.get(), tcl_sz.get());
  const Point end(tcl_ex.get(), tcl_ey.get(), tcl_ez.get());
  
  LatticeGeom *geom;

  if (geomtype.get() == 2) {
    geom = new IrregLatticeGeom(x, y, z, start, end);
  } else {
    geom = new LatticeGeom(x, y, z, start, end);
  }

  // Create attrib.
  const int mattribtype = attribtype.get();
  DiscreteAttrib<double> *attrib;
  switch (mattribtype)
    {
    case 4:
      attrib = new DiscreteAttrib<double>(x, y, z);
      break;
      
    case 3:
      attrib = new BrickAttrib<double>(x, y, z);
      break;

    case 2:
      attrib = new AccelAttrib<double>(x, y, z);
      break;
      
    default:
      attrib = new FlatAttrib<double>(x, y, z);
    }

  // Populate attrib, with analytic attrib and geometry.
  fill(attrib, geom, analyt);

  dbg << "Attrib in Genfield:\n" << attrib->getInfo() << endl;

  dbg << "Geometry in in Genfield:\n" << geom->getInfo() << endl;
  geom->getUnscaledTransform().print();
  geom->getUnscaledTransform().printi();
  ((Transform &)(geom->getTransform())).print();
  ((Transform &)(geom->getTransform())).printi();


  // Create index mapping.
  GenSField<double, LatticeGeom> *isf =
    new GenSField<double, LatticeGeom>(geom, attrib);
  double minval, maxval;
  isf->get_minmax(minval, maxval);

  if (indexed.get())
    {
      // Convert attrib -> indexed attrib.
      IndexAttrib<double, unsigned char, AccelAttrib<double> > *iatt =
	new IndexAttrib<double, unsigned char, AccelAttrib<double> >(x, y, z);

      indicize(iatt, attrib, minval, maxval, x, y, z);

      dbg << "Indexed Attrib in Genfield:\n" << iatt->getInfo() << endl;

      // Create Field, send it.
      GenSField<double, LatticeGeom> *osf =
	new GenSField<double, LatticeGeom>(geom, iatt);

      FieldHandle *hndl = new FieldHandle(osf);
      ofield->send(*hndl);
    }
  else
    {
      // Create Field, send it.
      GenSField<double, LatticeGeom> *osf =
	new GenSField<double, LatticeGeom>(geom, attrib);

      FieldHandle *hndl = new FieldHandle(osf);
      ofield->send(*hndl);
    }
}



void
GenField::send_vector_field()
{
  // get scalar, vector, tensor:
  vector<string> functs;

  string s = fval1.get()();
  functs.push_back(s);

  s = fval2.get()();
  functs.push_back(s);

  s = fval3.get()();
  functs.push_back(s);


  // Create analytic attrib.
  // TODO: error checking here?
  AnalytAttrib<Vector> *analyt = scinew AnalytAttrib<Vector>();
  analyt->set_fasteval();
  analyt->set_function(functs);

  // Create geometry.
  const int x = nx.get();
  const int y = ny.get();
  const int z = nz.get();

  const Point start(tcl_sx.get(), tcl_sy.get(), tcl_sz.get());
  const Point end(tcl_ex.get(), tcl_ey.get(), tcl_ez.get());
  
  LatticeGeom *geom = new LatticeGeom(x, y, z, start, end);

  // Create attrib.
  const int mattribtype = attribtype.get();
  DiscreteAttrib<Vector> *attrib;
  switch (mattribtype)
    {
    case 4:
      attrib = new DiscreteAttrib<Vector>(x, y, z);
      break;
      
    case 3:
      attrib = new BrickAttrib<Vector>(x, y, z);
      break;

    case 2:
      attrib = new AccelAttrib<Vector>(x, y, z);
      break;
      
    default:
      attrib = new FlatAttrib<Vector>(x, y, z);
    }

  // Populate attrib, with analytic attrib and geometry.
  fill(attrib, geom, analyt);

  dbg << "Attrib in Genfield:\n" << attrib->getInfo() << endl;

  // Create Field, send it.
  GenVField<Vector, LatticeGeom> *osf =
    new GenVField<Vector, LatticeGeom>(geom, attrib);

  FieldHandle *hndl = new FieldHandle(osf);
  ofield->send(*hndl);
}



void
GenField::send_tensor_field()
{
#if 0
  // get scalar, vector, tensor:
  vector<string> functs;

  string s = fval1.get()();
  functs.push_back(s);
  functs.push_back(s);
  functs.push_back(s);

  s = fval2.get()();
  functs.push_back(s);
  functs.push_back(s);
  functs.push_back(s);

  s = fval3.get()();
  functs.push_back(s);
  functs.push_back(s);
  functs.push_back(s);


  // Create analytic attrib.
  // TODO: error checking here?
  AnalytAttrib<Vector> *analyt = scinew AnalytAttrib<Tensor>();
  analyt->set_fasteval();
  analyt->set_function(functs);

  // Create geometry.
  const int x = nx.get();
  const int y = ny.get();
  const int z = nz.get();

  const Point start(tcl_sx.get(), tcl_sy.get(), tcl_sz.get());
  const Point end(tcl_ex.get(), tcl_ey.get(), tcl_ez.get());
  
  LatticeGeom *geom = new LatticeGeom(x, y, z, start, end);

  // Create attrib.
  const int mattribtype = attribtype.get();
  DiscreteAttrib<Tensor> *attrib;
  switch (mattribtype)
    {
    case 4:
      attrib = new DiscreteAttrib<Tensor>(x, y, z);
      break;
      
    case 3:
      attrib = new BrickAttrib<Tensor>(x, y, z);
      break;

    case 2:
      attrib = new AccelAttrib<Tensor>(x, y, z);
      break;
      
    default:
      attrib = new FlatAttrib<Tensor>(x, y, z);
    }

  // Populate attrib, with analytic attrib and geometry.
  fill(attrib, geom, analyt);

  dbg << "Attrib in Genfield:\n" << attrib->getInfo() << endl;

  // Create Field, send it.
  GenVField<Tensor, LatticeGeom> *osf =
    new GenVField<Tensor, LatticeGeom>(geom, attrib);

  FieldHandle *hndl = new FieldHandle(osf);
  ofield->send(*hndl);
#endif
}




void
GenField::execute()
{
  clString tclRes;
  TCL::eval(".ui"+id+".f.r.functions view", tclRes);

  if (tclRes == "2")
    {
      send_tensor_field();
    }
  else if (tclRes == "1")
    {
      send_vector_field();
    }
  else
    {
      send_scalar_field();
    }
}



} // End namespace Modules
} // End namespace PSECommon






