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
#include <SCICore/Datatypes/RegLatticeGeom.h>
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
  fval("fval", id, this), dbg("GenField", true){
  ofield=new SFieldOPort(this, "Geometry", SFieldIPort::Atomic);
  add_oport(ofield);

  RegLatticeGeom *geom = new RegLatticeGeom();
  FlatSAttrib<double> *attrib = new FlatSAttrib<double>();
  fldHandle = new GenSField<double>(geom, attrib);
 

}

GenField::~GenField()
  {
  }
    
void GenField::execute()
  {
    double mnx, mny, mnz;
    double retd;
    clString mfval, retval;
    char procstr[1000];
    mnx=nx.get()-1;
    mny=ny.get()-1;
    mnz=nz.get()-1;
    mfval=fval.get();

    GenSField<double> *osf = dynamic_cast<GenSField<double> *>(fldHandle.get_rep());
    osf->resize(mnx, mny, mnz);
    
    for(int i=0; i < mnx; i++){
      for(int j=0; j < mny; j++){
	for(int k=0; k < mnz; k++){
	  // set the values of x, y, and z; normalize to range from 0 to 1
	  sprintf(procstr, "set x %f; set y %f; set z %f",
		  i/mnx, j/mny, k/mnz);
	  TCL::eval(procstr, retval);
	  sprintf(procstr, "expr %s", mfval());
	  int err = TCL::eval(procstr, retval);
	  if(err){
	    return;
	  }
	  retval.get_double(retd);
	  osf->grid(Point(i, j, k)) = retd;
	}
      }
    }
    // set initial position to orgin and send field to out port
    osf->set_bounds(Point(0, 0, 0), Point(mnx, mny, mnz));
    ofield->send(osf);
  }

} // End namespace Modules
} // End namespace PSECommon

