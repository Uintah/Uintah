//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
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
  TetVolField<double> *tv = scinew TetVolField<double>(tvm, Field::NODE);
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
