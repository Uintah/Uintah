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
#include <SCICore/Datatypes/FlatSAttrib.h>
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
public:
  //      DebugStream dbg;
      
  TCLint nx;      // determines the size of the field
  TCLint ny;
  TCLint nz;
  TCLstring fval; // the equation to evaluate at each point in the field
  TCLint geomtype;
  
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
  dbg("GenField", true), fldHandle(){
  ofield=new SFieldOPort(this, "SField", SFieldIPort::Atomic);
  add_oport(ofield);
  
}

GenField::~GenField()
  {
  }
    
void GenField::execute()
  {
    double mnx, mny, mnz;
    double retd;
    clString mfval, retval;
    int mgeomtype;
    char procstr[1000];
    // Get the dimensions, subtract 1 for use in the inner loop
    mnx=nx.get()-1;
    mny=ny.get()-1;
    mnz=nz.get()-1;
    mfval=fval.get();
    mgeomtype = geomtype.get();
    dbg << "geomtype: " << mgeomtype << endl; 
    if(mgeomtype == 1){
      LatticeGeom *geom = new LatticeGeom();
      if(!geom){
	error("Failed new: new LatticeGeom");
	return;
      }
      FlatSAttrib<double> *attrib = new FlatSAttrib<double>();
      if(!attrib){
	error("Failed new: new FlatSAttrib");
	return;
      }
      GenSField<double, LatticeGeom> *osf = new GenSField<double, LatticeGeom>(geom, attrib);
      if(!osf){
	error("Failed new: new GenSField");
	return;
      }
      osf->set_bbox(Point(0, 0, 0), Point(mnx, mny, mnz));
      osf->resize(mnx+1, mny+1, mnz+1);
      for(int k=0; k <= mnx; k++){
	for(int j=0; j <= mny; j++){
	  for(int i=0; i <= mnz; i++){
	    // set the values of x, y, and z; normalize to range from 0 to 1
	    sprintf(procstr, "set x %f; set y %f; set z %f",
		    i/mnx, j/mny,  k/mnz);
	    TCL::eval(procstr, retval);
	    sprintf(procstr, "expr %s", mfval());
	    int err = TCL::eval(procstr, retval);
	    if(err){
	      // Error in evaluation of user defined function, give warning and return
	      error("Error in evaluation of function defined by user in GenField");
	      return;
	    }
	    retval.get_double(retd);
	    attrib->grid(i, j, k) = retd;
	  }
	}
      }
      ofield->send(osf);
    }
    else if(mgeomtype == 2){
      MeshGeom *geom = new MeshGeom();
      if(!geom){
	error("Failed new: new MeshGeom");
	return;
      }
      FlatSAttrib<double> *attrib = new FlatSAttrib<double>();
      if(!attrib){
	error("Failed new: new FlatSAttrib");
	return;
      }
      GenSField<double, MeshGeom> *osf = new GenSField<double, MeshGeom>(geom, attrib);
      if(!osf){
	error("Failed new: new GenSField");
	return;
      }
      vector<Node> nodes(mnx*mny*mnz);
      vector<Tetrahedral> tets;
      int l = 0;
      for(int k=0; k <= mnx; k++){
	for(int j=0; j <= mny; j++){
	  for(int i=0; i <= mnz; i++){
	    // create the point at (i, j, k)
	    nodes[l++].p = Point(i, j, k);
	    // Assign the value of 
	    
	  }
	}
      }
      ofield->send(osf);
    }
  }

} // End namespace Modules
} // End namespace PSECommon

