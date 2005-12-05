//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : SinglePrism.cc
//    Author : Martin Cole
//    Date   : Mon Dec  5 13:23:20 2005

#include <Core/Basis/PrismLinearLgn.h>
#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  typedef PrismVolMesh<PrismLinearLgn<Point> > PVMesh;
  PVMesh *pvm = new PVMesh();

  Point p1(0.,0.,0.);
  Point p2(1.,0.,0.);
  Point p3(0.,1.,0.);
  Point p4(0.,0.,1.);
  Point p5(1.,0.,1.);
  Point p6(0.,1.,1.);


  pvm->add_point(p1);
  pvm->add_point(p2);
  pvm->add_point(p3);
  pvm->add_point(p4);
  pvm->add_point(p5);
  pvm->add_point(p6);

  pvm->add_prism(0, 1, 2, 3, 4, 5);

  typedef PrismLinearLgn<double>  DatBasis;
  typedef GenericField<PVMesh, DatBasis, vector<double> > PVField;
  PVField *pv = scinew PVField(pvm);
  pv->resize_fdata();

  const double div = 1./6.;
  PVField::fdata_type &d = pv->fdata();
  for (int i = 0; i < 6; ++i) {
    d[i] = div*i;
  }

  FieldHandle pvH(pv);
  TextPiostream out_stream("SinglePrism.pvd.fld", Piostream::Write);
  Pio(out_stream, pvH);

  return 0;  
}    
