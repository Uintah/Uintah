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

#include <SCICore/Datatypes/SField.h>
#include <SCICore/Datatypes/GenSField.h>
#include <SCICore/Datatypes/LatticeGeom.h>
#include <SCICore/Datatypes/MeshGeom.h>
#include <SCICore/Datatypes/MeshGeom.h>
#include <SCICore/Datatypes/FlatAttrib.h>
#include <SCICore/Datatypes/AccelAttrib.h>
#include <SCICore/Datatypes/BrickAttrib.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Datatypes/SFieldPort.h>
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
  SFieldOPort* ofield;
  DebugStream dbg;

  template <class A> void tfill(A *attrib, int x, int y, int z);
  template <class T> void fill(DiscreteAttrib<T> *attrib, int x, int y, int z);

public:
  //      DebugStream dbg;
      
  TCLint nx;      // determines the size of the field
  TCLint ny;
  TCLint nz;
  TCLstring fval; // the equation to evaluate at each point in the field
  TCLint geomtype;
  TCLint attribtype;
  
  // constructor, destructor:
  GenField(const clString& id);
  virtual ~GenField();
  SFieldHandle fldHandle;
  virtual void execute();
};

extern "C" Module* make_GenField(const clString& id){
  return new GenField(id);
}

GenField::GenField(const clString& id)
  : Module("GenField", id, Filter), 
    nx("nx", id, this), ny("ny", id, this), nz("nz", id, this), 
    fval("fval", id, this), geomtype("geomtype", id, this),
    attribtype("attribtype", id, this),
    dbg("GenField", true), fldHandle()
{
  ofield=new SFieldOPort(this, "SField", SFieldIPort::Atomic);
  add_oport(ofield);
  geomtype.set(1);
  attribtype.set(1);
}

GenField::~GenField()
{
}


template <class T>
void GenField::fill(DiscreteAttrib<T> *attrib, int x, int y, int z)
{
  clString mfval, retval;
  mfval=fval.get();

  char procstr[1000];
  const double mx = x - 1.0;
  const double my = y - 1.0;
  const double mz = z - 1.0;
  for(int k=0; k < z; k++) {
    for(int j=0; j < y; j++) {
      for(int i=0; i < x; i++) {
	// Set the values of x, y, and z; normalize to range from 0 to 1.
	sprintf(procstr, "set x %f; set y %f; set z %f",
		i/mx, j/my,  k/mz);
	TCL::eval(procstr, retval);
	sprintf(procstr, "expr %s", mfval());
	int err = TCL::eval(procstr, retval);
	if(err)
	  {
	    // Error in evaluation of user defined function, give
	    // warning and return.
	    error("Error in evaluation of user defined function in GenField");
	    return;
	  }
	double retd = 0.0;
	retval.get_double(retd);
	attrib->set3(i, j, k, (T)retd);
      }
    }
  }
}


template <class A>
void GenField::tfill(A *attrib, int x, int y, int z)
{
  clString mfval, retval;
  mfval=fval.get();

  char procstr[1000];
  const double mx = x - 1.0;
  const double my = y - 1.0;
  const double mz = z - 1.0;
  for(int k=0; k < z; k++) {
    for(int j=0; j < y; j++) {
      for(int i=0; i < x; i++) {
	// Set the values of x, y, and z; normalize to range from 0 to 1.
	sprintf(procstr, "set x %f; set y %f; set z %f",
		i/mx, j/my,  k/mz);
	TCL::eval(procstr, retval);
	sprintf(procstr, "expr %s", mfval());
	int err = TCL::eval(procstr, retval);
	if(err)
	  {
	    // Error in evaluation of user defined function, give
	    // warning and return.
	    error("Error in evaluation of user defined function in GenField");
	    return;
	  }
	double retd = 0.0;
	retval.get_double(retd);
	attrib->fset3(i, j, k, retd);
      }
    }
  }
}

    
void
GenField::execute()
{
  // Get the dimensions, subtract 1 for use in the inner loop
  const int x = nx.get();
  const int y = ny.get();
  const int z = nz.get();

  const int mgeomtype = geomtype.get();
  const int mattribtype = attribtype.get();

  //if (mgeomtype != 1) { return; }

  LatticeGeom *geom = new LatticeGeom();
  geom->resize(x, y, z);

  dbg << "attribtype: " << mattribtype << endl; 
  //switch (mattribtype)
  DiscreteAttrib<double> *attrib;
#if 0
  switch (mgeomtype)
    {
    case 4:
      attrib = new DiscreteAttrib<double>(x, y, z);
      tfill((DiscreteAttrib<double> *)attrib, x, y, z);
      break;
      
    case 3:
      attrib = new BrickAttrib<double>(x, y, z);
      tfill((BrickAttrib<double> *)attrib, x, y, z);
      break;

    case 2:
      attrib = new AccelAttrib<double>(x, y, z);
      tfill((AccelAttrib<double> *)attrib, x, y, z);
      break;
      
    default:
      attrib = new FlatAttrib<double>(x, y, z);
      tfill((FlatAttrib<double> *)attrib, x, y, z);
    }
#elif 0
  attrib = new WrapAttrib<double>(x, y, z);
  tfill((WrapAttrib<double> *)attrib, x, y, z);
  dbg << "Attrib in Genfield:\n" << attrib->get_info() << endl;
#else
  attrib = new BrickAttrib<double>(x, y, z);
  tfill((BrickAttrib<double> *)attrib, x, y, z);
  dbg << "Attrib in Genfield:\n" << attrib->get_info() << endl;
#endif

  //dbg << "filling" << endl;
  //fill(attrib, x, y, z);
  GenSField<double, LatticeGeom> *osf =
    new GenSField<double, LatticeGeom>(geom, attrib);
  osf->set_bbox(Point(0, 0, 0), Point(x-1, y-1, z-1));
  SFieldHandle *hndl = new SFieldHandle(osf);
  ofield->send(*hndl);
}


} // End namespace Modules
} // End namespace PSECommon

