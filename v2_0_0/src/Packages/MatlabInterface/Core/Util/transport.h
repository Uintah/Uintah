/*
 *  Matlab.cc:
 *
 *  Written by:
 *   oleg
 *   TODAY'S DATE HERE
 *
 */

#ifndef MatlabInterface_Core_Util_transport_h
#define MatlabInterface_Core_Util_transport_h

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <Packages/MatlabInterface/Core/Util/bring.h>

#include <Packages/MatlabInterface/share/share.h>
#include <stdio.h>

namespace MatlabInterface {

using namespace SCIRun;

MatrixHandle transport(int wordy,int flag,const char *hport, MatrixHandle mh);
void         transport(int wordy,int flag,const char *hport, char *cmd);

void trnsp(double *p1,double *p2,int n,int m);
void prt(double *m,int n1,int n2);



}

#endif
