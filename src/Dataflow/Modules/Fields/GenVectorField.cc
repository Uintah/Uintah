
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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Datatypes/VectorField.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Thread/Time.h>

#include <Core/Math/function.h>


namespace SCIRun {
    

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
  //using Pio;
    
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
  vfd = scinew VectorFieldRG(_resx, _resy, _resz);
  handle = vfd;
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

} // End namespace SCIRun

