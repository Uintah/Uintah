/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

//    File   : SingleTet.cc
//    Author : Martin Cole
//    Date   : Thu Feb 28 17:09:21 2002


#include <Core/Datatypes/TetVolField.h>
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
  TetVolMesh *tvm = new TetVolMesh();

  Point p1(0.,0.,0.);
  Point p2(1.,0.,0.);
  Point p3(0.,1.,0.);
  Point p4(0.,0.,1.);
  tvm->add_point(p1);
  tvm->add_point(p2);
  tvm->add_point(p3);
  tvm->add_point(p4);

  tvm->add_tet(0, 1, 2, 3);
  TetVolField<double> *tv = scinew TetVolField<double>(tvm, 1);
  tv->resize_fdata();

  TetVolField<double>::fdata_type &d = tv->fdata();
  d[0] = 0.0;
  d[1] = 0.25;
  d[2] = 0.75;
  d[3] = 1.0;

  FieldHandle tvH(tv);
  TextPiostream out_stream("SingleTet.tvd.fld", Piostream::Write);
  Pio(out_stream, tvH);

  return 0;  
}    
