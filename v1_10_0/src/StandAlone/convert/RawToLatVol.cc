/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  RawToLatVolField.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/LatVolField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;
using std::cin;
using std::cout;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: "<<argv[0]<<" raw lattice\n";
    return 0;
  }

  cout << "What are the dimensions: nx ny nz? ";
  int ni, nj, nk;
  double minx, miny, minz, maxx, maxy, maxz;
  cin >> ni >> nj >> nk;
  cout << "ASCII or Binary input (a/b)? ";
  string ascii_or_binary_input;
  cin >> ascii_or_binary_input;
  cout << "Datatype (d/f/ui/i/us/s/uc/c)? ";
  string datatype;
  cin >> datatype;
  cout << "Min x y z? ";
  cin >> minx >> miny >> minz;
  cout << "Max x y z? ";
  cin >> maxx >> maxy >> maxz;
  cout << "Data at nodes or cells (n/c)? ";
  string nodes_or_cells;
  cin >> nodes_or_cells;
  cout << "ASCII or Binary output (a/b)? ";
  string ascii_or_binary_output;
  cin >> ascii_or_binary_output;

  Point min(minx, miny, minz), max(maxx, maxy, maxz);
  LatVolMeshHandle lvm;
  if (nodes_or_cells == "n") lvm = new LatVolMesh(ni, nj, nk, min, max);
  else lvm = new LatVolMesh(ni+1, nj+1, nk+1, min, max);

  FieldHandle fH;

  FILE *fin = fopen(argv[1], "rt");

  if (datatype == "d") {
    LatVolField<double> *lv;
    if (nodes_or_cells == "n") 
      lv = new LatVolField<double>(lvm, Field::NODE);
    else 
      lv = new LatVolField<double>(lvm, Field::CELL);
    fH=lv;
    double *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary_input == "a") {
      double f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%lf", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(double), ni*nj*nk, fin);
    }
  } else if (datatype == "f") {
    LatVolField<float> *lv;
    if (nodes_or_cells == "n") 
      lv = new LatVolField<float>(lvm, Field::NODE);
    else 
      lv = new LatVolField<float>(lvm, Field::CELL);
    fH=lv;
    float *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary_input == "a") {
      float f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%f", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(float), ni*nj*nk, fin);
    }
  } else if (datatype == "ui") {
    LatVolField<unsigned int> *lv;
    if (nodes_or_cells == "n") 
      lv = new LatVolField<unsigned int>(lvm, Field::NODE);
    else
      lv = new LatVolField<unsigned int>(lvm, Field::CELL);
    fH=lv;
    unsigned int *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary_input == "a") {
      unsigned int f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%u", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(unsigned int), ni*nj*nk, fin);
    }
  } else if (datatype == "i") {
    LatVolField<int> *lv;
    if (nodes_or_cells == "n") 
      lv = new LatVolField<int>(lvm, Field::NODE);
    else
      lv = new LatVolField<int>(lvm, Field::CELL);
    fH=lv;
    int *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary_input == "a") {
      int f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%d", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(int), ni*nj*nk, fin);
//      char *d = new char[ni*nj*nk];
//      fread(d, sizeof(char), ni*nj*nk, fin);
//      for (int i=0; i<ni*nj*nk; i++)
//	*data++ = (int)d[i];
//      delete[] d;
    }
  } else if (datatype == "us") {
    LatVolField<unsigned short> *lv;
    if (nodes_or_cells == "n") 
      lv = new LatVolField<unsigned short>(lvm, Field::NODE);
    else
      lv = new LatVolField<unsigned short>(lvm, Field::CELL);
    fH=lv;
    unsigned short *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary_input == "a") {
      unsigned short f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%hu", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(unsigned short), ni*nj*nk, fin);
    }
  } else if (datatype == "s") {
    LatVolField<short> *lv;
    if (nodes_or_cells == "n") 
      lv = new LatVolField<short>(lvm, Field::NODE);
    else
      lv = new LatVolField<short>(lvm, Field::CELL);
    fH=lv;
    short *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary_input == "a") {
      short f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%hd", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(short), ni*nj*nk, fin);
    }
  } else if (datatype == "uc") {
    LatVolField<unsigned char> *lv;
    if (nodes_or_cells == "n") 
      lv = new LatVolField<unsigned char>(lvm, Field::NODE);
    else
      lv = new LatVolField<unsigned char>(lvm, Field::CELL);
    fH=lv;
    unsigned char *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary_input == "a") {
      unsigned short f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%hu", &f);
	*data++ = (unsigned char)f;
      }
    } else {
      fread(data, sizeof(unsigned char), ni*nj*nk, fin);
    }
  } else if (datatype == "c") {
    LatVolField<char> *lv;
    if (nodes_or_cells == "n") 
      lv = new LatVolField<char>(lvm, Field::NODE);
    else
      lv = new LatVolField<char>(lvm, Field::CELL);
    fH=lv;
    char *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary_input == "a") {
      short f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%hd", &f);
	*data++ = (char)f;
      }
    } else {
      fread(data, sizeof(char), ni*nj*nk, fin);
    }
  } else {
    cerr << "Error -- unknown datatype: "<<datatype<<"\n";
  }
  if (ascii_or_binary_output == "a") {
    TextPiostream out_stream(argv[2], Piostream::Write);
    Pio(out_stream, fH);
  } else if (ascii_or_binary_output == "b") {
    BinaryPiostream out_stream(argv[2], Piostream::Write);
    Pio(out_stream, fH);
  } else {
    cerr << "Error -- unknown ASCII/binary flag: "<<ascii_or_binary_output<<"\n";
  }
  return 0;  
}    
