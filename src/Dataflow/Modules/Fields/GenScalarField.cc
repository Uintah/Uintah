
/*
 *  GenScalarField.cc: ScalarField Generator class
 *
 *  Written by:
 *   David Hart
 *   Department of Computer Science
 *   University of Utah
 *   Copyright (C) 2000 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/function.h>

//#include <iostream.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Math;

//----------------------------------------------------------------------
class GenScalarField : public Module {
  
  ScalarFieldOPort* outport;
  ScalarFieldHandle handle;

  //TCLdouble r1, r2;
  //TCLdouble x1, y1, z1, x2, y2, z2;
  TCLint resx, resy, resz;
  TCLstring eqn;
  TCLdouble xmin, xmax, ymin, ymax, zmin, zmax;

public:
  
  GenScalarField(const clString& id);
  virtual ~GenScalarField();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

//----------------------------------------------------------------------
extern "C" Module* make_GenScalarField(const clString& id) {
  return new GenScalarField(id);
}

//----------------------------------------------------------------------
static clString module_name("GenScalarField");

//----------------------------------------------------------------------
GenScalarField::GenScalarField(const clString& id)
  : Module("GenScalarField", id, Source),
  resx("resx", id, this),
  resy("resy", id, this),
  resz("resz", id, this),
  xmin("xmin", id, this),
  ymin("ymin", id, this),
  zmin("zmin", id, this),
  xmax("xmax", id, this),
  ymax("ymax", id, this),
  zmax("zmax", id, this),
  eqn("eqn", id, this)
{
				// Create the output data handle and port
  outport=scinew ScalarFieldOPort(this, "Output Data",
    ScalarFieldIPort::Atomic);

  add_oport(outport);
}

//----------------------------------------------------------------------
GenScalarField::~GenScalarField()
{
  ScalarField* sf = handle.get_rep();
  if (!sf) delete sf;
  sf = NULL;
}

//----------------------------------------------------------------------
void GenScalarField::execute()
{

  //static int oldres;
  int _resx, _resy, _resz;
  clString _eqn;
  double _xmin, _xmax, _ymin, _ymax, _zmin, _zmax;
  
  ScalarField* sf = handle.get_rep();
  ScalarFieldRGdouble* sfd = (ScalarFieldRGdouble*) sf;

  _resx = resx.get();
  _resy = resy.get();
  _resz = resz.get();

  _xmin = xmin.get();
  _ymin = ymin.get();
  _zmin = zmin.get();
  _xmax = xmax.get();
  _ymax = ymax.get();
  _zmax = zmax.get();

  _eqn = eqn.get();

  double x[3];
  
  int i, j, k;

  cout << "deleteing old grid" << endl;
  if (!sf) delete sf;
  sf = NULL;

  cout << "allocating scalar grid" << endl;
  cout << "res: " << _resx << "x" << _resy << "x" << _resz << endl;
  cout << "x: (" << _xmin << "," << _xmax << ")" << endl;
  cout << "y: (" << _ymin << "," << _ymax << ")" << endl;
  cout << "z: (" << _zmin << "," << _zmax << ")" << endl;
  sfd = scinew ScalarFieldRGdouble();
  handle = sfd;
  
  sfd->resize(_resx,_resy,_resz);
  sfd->set_bounds(Point(_xmin, _ymin, _zmin),
    Point(_xmax, _ymax, _zmax));

  double dx = (_xmax-_xmin) / double(_resx);
  double dy = (_ymax-_ymin) / double(_resy);
  double dz = (_zmax-_zmin) / double(_resz);

  Function* f = NULL;
  fnparsestring(_eqn(), &f);
  if (!f) f = new Function(0.0);
  cout << "populating scalar grid" << endl;
  FuncEvalPtr eval = f->getFastEval();
				// make sure that eval was
				// successfully generated
  if (eval) {
    for (x[0] = _xmin, i = 0; i < _resx; x[0] += dx, i++) {
      cout << "." << flush;
      for (x[1] = _ymin, j = 0; j < _resy; x[1] += dy, j++) {
	for (x[2] = _zmin, k = 0; k < _resz; x[2] += dz, k++) {
	  sfd->grid(i, j, k) = eval(x);
	}
      }
    }
  }
				// if fast eval wasn't created, fall
				// back to the slow but sure version
  else {
    for (x[0] = _xmin, i = 0; i < _resx; x[0] += dx, i++) {
      cout << "." << flush;
      for (x[1] = _ymin, j = 0; j < _resy; x[1] += dy, j++) {
	for (x[2] = _zmin, k = 0; k < _resz; x[2] += dz, k++) {
	  sfd->grid(i, j, k) = f->eval(x);
	}
      }
    }
  }

  cout << endl;
  
  sfd->compute_bounds();
  sfd->compute_minmax();

  delete f;
    
  cout << "sending scalar grid" << endl;
  outport->send(handle);
}

//----------------------------------------------------------------------
void GenScalarField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


} // End namespace Modules
} // End namespace PSECommon

//----------------------------------------------------------------------
//
//
