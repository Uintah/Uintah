
/* GenVectorField.cc: generate a vector field from a system of
 *  equations
 *
 *  Written by:
 *   David Hart
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Datatypes/VectorField.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Time.h>

#include <SCICore/Math/function.h>

using namespace SCICore::Datatypes;
using namespace SCICore::Math;
using namespace SCICore::Thread;

namespace PSECommon {
namespace Modules {
    
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

//----------------------------------------------------------------------
class GenVectorField : public Module {

protected:
  VectorFieldOPort* outport;
  VectorFieldHandle handle;
  
  //TCLstring filename;
  //clString old_filename;

  TCLint resx, resy, resz;
  TCLstring eqX, eqY, eqZ;
  TCLdouble xmin, xmax, ymin, ymax, zmin, zmax;
  
public:
  
  GenVectorField(const clString& id);
  virtual ~GenVectorField();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
  
};

//----------------------------------------------------------------------
extern "C" Module* make_GenVectorField(const clString& id) {
  return new GenVectorField(id);
}

//----------------------------------------------------------------------
static clString module_name("GenVectorField");

//----------------------------------------------------------------------
GenVectorField::GenVectorField(const clString& id)
  : Module("GenVectorField", id, Source),
  //filename("filename", id, this),
  resx("resx", id, this),
  resy("resy", id, this),
  resz("resz", id, this),
  xmin("xmin", id, this),
  ymin("ymin", id, this),
  zmin("zmin", id, this),
  xmax("xmax", id, this),
  ymax("ymax", id, this),
  zmax("zmax", id, this),
  eqY("eqY", id, this),
  eqX("eqX", id, this),
  eqZ("eqZ", id, this)
{
				// Create the output data handle and
				// port
  outport=scinew VectorFieldOPort(this, "Output Data",
    VectorFieldIPort::Atomic);
  add_oport(outport);
}

//----------------------------------------------------------------------
GenVectorField::~GenVectorField()
{
  VectorField* vf = handle.get_rep();
  if (!vf) delete vf;
  vf = NULL;
}

//----------------------------------------------------------------------
void GenVectorField::execute()
{
  //using SCICore::Containers::Pio;
    
  int _resx, _resy, _resz;
  clString _eqX, _eqY, _eqZ;
  double _xmin, _xmax, _ymin, _ymax, _zmin, _zmax;
    
  VectorField* vf = handle.get_rep();
  VectorFieldRG* vfd = (VectorFieldRG*) vf;

  Function* fx = NULL;
  Function* fy = NULL;
  Function* fz = NULL;

  _resx = resx.get();
  _resy = resy.get();
  _resz = resz.get();

  _xmin = xmin.get();
  _ymin = ymin.get();
  _zmin = zmin.get();
  _xmax = xmax.get();
  _ymax = ymax.get();
  _zmax = zmax.get();

  _eqX = eqX.get();
  _eqY = eqY.get();
  _eqZ = eqZ.get();

  cout << _eqX << endl;
  fnparsestring(_eqX(), &fx);
  if (!fx) fx = new Function(0.0);

  cout << _eqY << endl;
  fnparsestring(_eqY(), &fy);
  if (!fy) fy = new Function(0.0);

  cout << _eqZ << endl;
  fnparsestring(_eqZ(), &fz);
  if (!fz) fz = new Function(0.0);
  
  cout << "x := " << fx << endl;
  cout << "y := " << fy << endl;
  cout << "z := " << fz << endl;

  double x[3];
  int i, j, k;

  cout << "deleting old grid" << endl;
  if (!vf) delete vf;
  vf = NULL;

  cout << "allocating vector grid" << endl;
  cout << "res: " << _resx << "x" << _resy << "x" << _resz << endl;
  cout << "x: (" << _xmin << "," << _xmax << ")" << endl;
  cout << "y: (" << _ymin << "," << _ymax << ")" << endl;
  cout << "z: (" << _zmin << "," << _zmax << ")" << endl;
  vfd = scinew VectorFieldRG;
  handle = vfd;
  vfd->resize(_resx,_resy,_resz);
  vfd->set_bounds(Point(_xmin, _ymin, _zmin),
    Point(_xmax, _ymax, _zmax));

  double dx = (_xmax-_xmin) / double(_resx);
  double dy = (_ymax-_ymin) / double(_resy);
  double dz = (_zmax-_zmin) / double(_resz);

  FuncEvalPtr evalx = fx->getFastEval();
  FuncEvalPtr evaly = fy->getFastEval();
  FuncEvalPtr evalz = fz->getFastEval();

  cout << "populating vector grid" << endl;
  
				// make sure that all eval functions
				// were successfully generated
  if (evalx && evaly && evalz) {
    for (x[0] = _xmin, i = 0; i < _resx; x[0] += dx, i++) {
      cout << "." << flush;
      for (x[1] = _ymin, j = 0; j < _resy; x[1] += dy, j++) {
	for (x[2] = _zmin, k = 0; k < _resz; x[2] += dz, k++) {
	  vfd->grid(i, j, k) = Vector(evalx(x),evaly(x),evalz(x));
	}
      }
    }
  }
  else {
				// if any of the fast eval functions
				// weren't created, fall back to the
				// slow but sure version
    for (x[0] = _xmin, i = 0; i < _resx; x[0] += dx, i++) {
      cout << "." << flush;
      for (x[1] = _ymin, j = 0; j < _resy; x[1] += dy, j++) {
	for (x[2] = _zmin, k = 0; k < _resz; x[2] += dz, k++) {
	  vfd->grid(i, j, k) = Vector(fx->eval(x),fx->eval(x),fx->eval(x));
	}
      }
    }
  }

  cout << endl;

  vfd->compute_bounds();

  delete fx;
  delete fy;
  delete fz;

  cout << "sending vector grid" << endl;
  outport->send(handle);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
void GenVectorField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  2000/07/23 18:30:11  dahart
// Initial commit / Modules to generate scalar & vector fields from
// symbolic functions
//
// Revision 1.6  2000/03/17 09:27:13  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/08/25 03:47:55  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:52  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:52  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:36  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:49  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:27  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:54  dav
// updates in Modules for Datatypes
//
// Revision 1.1  1999/04/25 02:38:11  dav
// more things that should have been there but were not
//
//
