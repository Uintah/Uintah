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
//    File   : SingleHex.cc
//    Author : Martin Cole
//    Date   : Sat Oct 29 09:51:40 2005

#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/HexVolMesh.h>
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
  typedef HexVolMesh<HexTrilinearLgn<Point> > HVMesh;
  HVMesh *hvm = new HVMesh();

  Point p1(0.,0.,0.);
  Point p2(1.,0.,0.);
  Point p3(1.,1.,0.);
  Point p4(0.,1.,0.);
  Point p5(0.,0.,1.);
  Point p6(1.,0.,1.);
  Point p7(1.,1.,1.);
  Point p8(0.,1.,1.);

  hvm->add_point(p1);
  hvm->add_point(p2);
  hvm->add_point(p3);
  hvm->add_point(p4);
  hvm->add_point(p5);
  hvm->add_point(p6);
  hvm->add_point(p7);
  hvm->add_point(p8);

  hvm->add_hex(0, 1, 2, 3, 4, 5, 6, 7);

  typedef HexTrilinearLgn<double>  DatBasis;
  typedef GenericField<HVMesh, DatBasis, vector<double> > HVField;
  HVField *hv = scinew HVField(hvm);
  hv->resize_fdata();

  const double div = 1./8.;
  HVField::fdata_type &d = hv->fdata();
  for (int i = 0; i < 8; ++i) {
    d[i] = div*i;
  }

  FieldHandle hvH(hv);
  TextPiostream out_stream("SingleHex.hvd.fld", Piostream::Write);
  Pio(out_stream, hvH);

  return 0;  
}    
