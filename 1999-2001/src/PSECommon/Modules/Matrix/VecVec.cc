
/*
 *  VecVec: Vector - Vector operations (e.g. addition, subtraction, ...)
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/SymSparseRowMatrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class VecVec : public Module {
  ColumnMatrixIPort* icol1;
  ColumnMatrixIPort* icol2;
  ColumnMatrixOPort* ocol;
  TCLstring opTCL;
public:
  VecVec(const clString& id);
  virtual ~VecVec();
  virtual void execute();
};

extern "C" Module* make_VecVec(const clString& id)
{
  return new VecVec(id);
}

VecVec::VecVec(const clString& id)
  : Module("VecVec", id, Filter), opTCL("opTCL", id, this)
{
  icol1=new ColumnMatrixIPort(this, "a", ColumnMatrixIPort::Atomic);
  add_iport(icol1);
  icol2=new ColumnMatrixIPort(this, "b", MatrixIPort::Atomic);
  add_iport(icol2);
  
  // Create the output port
  ocol=new ColumnMatrixOPort(this, "Output", ColumnMatrixIPort::Atomic);
  add_oport(ocol);
}

VecVec::~VecVec()
{
}

void VecVec::execute() {
  update_state(NeedData);
  int haveData1=1;
  int haveData2=1;
  int i;
  ColumnMatrixHandle icol1H;
  ColumnMatrixHandle icol2H;
  if (!icol1->get(icol1H) || !icol1H.get_rep()) haveData1=0;
  if (!icol2->get(icol2H) || !icol2H.get_rep()) haveData2=0;
  if (haveData1 && !haveData2) {
    ColumnMatrix *res = scinew ColumnMatrix(icol1H->nrows());
    for (i=0; i<icol1H->nrows(); i++)
      (*res)[i]=(*(icol1H.get_rep()))[i];
    cerr << "VecVec: sending input 1 through...\n";
    ocol->send(ColumnMatrixHandle(res));
    return;
  } else if (!haveData1 && haveData2) {
    ColumnMatrix *res = scinew ColumnMatrix(icol2H->nrows());
    for (i=0; i<icol2H->nrows(); i++)
      (*res)[i]=(*(icol2H.get_rep()))[i];
    cerr << "VecVec: sending input 2 through...\n";
    ocol->send(ColumnMatrixHandle(res));
    return;
  } else if (!haveData1 && !haveData2) {
    cerr << "VecVec: error - no data.\n";
    return;
  }

  clString opS=opTCL.get();
  update_state(JustStarted);
  ColumnMatrix *res;
  if (opS == "plus") {
    if (icol1H->nrows() != icol2H->nrows()) {
      cerr << "VecVec: error, column matrices must have same number of rows ("<<icol1H->nrows()<<"!="<<icol2H->nrows()<<")\n";
      return;
    }
    res = scinew ColumnMatrix(icol1H->nrows());
    for (i=0; i<icol1H->nrows(); i++)
      (*res)[i]=(*(icol1H.get_rep()))[i]+(*(icol2H.get_rep()))[i];
    ocol->send(ColumnMatrixHandle(res));
  } else if (opS == "minus") {
    if (icol1H->nrows() != icol2H->nrows()) {
      cerr << "VecVec: error, column matrices must have same number of rows ("<<icol1H->nrows()<<"!="<<icol2H->nrows()<<")\n";
      return;
    }
    res = scinew ColumnMatrix(icol1H->nrows());
    for (i=0; i<icol1H->nrows(); i++)
      (*res)[i]=(*(icol1H.get_rep()))[i]-(*(icol2H.get_rep()))[i];
    ocol->send(ColumnMatrixHandle(res));
  } else if (opS == "cat") {
    res = scinew ColumnMatrix(icol1H->nrows()+icol2H->nrows());
    for (i=0; i<icol1H->nrows(); i++)
      (*res)[i]=(*(icol1H.get_rep()))[i];
    for (i=0; i<icol2H->nrows(); i++) 
      (*res)[i+icol1H->nrows()]=(*(icol2H.get_rep()))[i];
    ocol->send(ColumnMatrixHandle(res));
  } else {
    cerr << "VecVec: unknown operation "<<opS<<"\n";
  }
}    
} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  2000/12/13 21:03:18  dmw
// Added concatination of vectors
//
// Revision 1.1  2000/11/02 21:43:32  dmw
// added VecVec module
//
