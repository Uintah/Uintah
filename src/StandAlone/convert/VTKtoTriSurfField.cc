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

//    File   : VTKtoTriSurfField.cc
//    Author : Martin Cole
//    Date   : Fri May  7 10:23:05 2004


// Warning!: this converter is only partially implemented. It supports only 
//           a subset of the vtk format. Please extend it as you need more

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <Core/Init/init.h>
#include <StandAlone/convert/FileUtils.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

#define check_error(str) \
  if (str.fail()) { \
    cerr << "fail state at line: " << __LINE__ << " " << endl; \
    return 66; }

bool bin_out;
bool swap_endian;


typedef TriLinearLgn<float>               LDatBasis;
typedef ConstantBasis<float>              CDatBasis;
typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;
typedef GenericField<TSMesh, LDatBasis,  vector<float> > TSFieldL;    
typedef GenericField<TSMesh, CDatBasis,  vector<float> > TSFieldC;    

void setDefaults() {
  bin_out=false;
  swap_endian=false;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg], "-bin_out")) {
      bin_out=true;
      currArg++;
    } else if (!strcmp(argv[currArg], "-swap_endian")) {
      swap_endian=true;
      currArg++;
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

void printUsageInfo(char *progName) {
  cerr << endl 
       << "Usage: " << progName 
       << " VTKFile TriSurfField [-bin_out] [-swap_endian]" << endl << endl
       << "\t This program will read in a .vtk file and output a TriSurfField"
       << endl
       << "\t -bin_out specifies binare vs. ASCII output. ASCII by default."
       << endl 
       << "\t -swap_endian swaps the data from big to little endian or" 
       << endl
       << "\t vice versa. false by default."
       << endl << endl;

  cerr << "Warning!: this converter is only partially implemented. It supports "
       << "only a subset of the vtk format. Please extend it as you need more."
       << endl;
}


void swap_endianess_4(unsigned *dw)
{
  if (!swap_endian) return;
  register unsigned tmp;
  tmp =  (*dw & 0x000000FF);
  tmp = ((*dw & 0x0000FF00) >> 0x08) | (tmp << 0x08);
  tmp = ((*dw & 0x00FF0000) >> 0x10) | (tmp << 0x08);
  tmp = ((*dw & 0xFF000000) >> 0x18) | (tmp << 0x08);
  *dw = tmp;
}

template <class T>
int
read_n_points(TSMesh *tsm, int n, ifstream &str) {
  T arr[3];
  check_error(str);

  for(int i = 0; i < n; i++) {
    str.read((char*)arr, sizeof(T) * 3);
    check_error(str);
    switch (sizeof(T)) {
    case 2 :

      break;
    case 4:
      swap_endianess_4((unsigned*)&arr[0]);
      swap_endianess_4((unsigned*)&arr[1]);
      swap_endianess_4((unsigned*)&arr[2]);
      break;
    case 8:

      break;
    }
    tsm->add_point(Point(arr[0], arr[1], arr[2]));
    //cout << arr[0] << ", " << arr[1] << ", " << arr[2] << endl;
  }
  return 0;
}

template <class T>
int
read_n_polys(TSMesh *tsm, int n, ifstream &str) {
  T arr[3];
  check_error(str);
  for(int i = 0; i < n; i++) {
    T val;
    str.read((char*)&val, sizeof(T));
    check_error(str);
    swap_endianess_4(&val);

    if (val != 3) {
      cout << "ERROR: can only handle triangle polys atm..." << endl;
      exit(1);
    }

    str.read((char*)arr, sizeof(T) * 3);
    check_error(str);

    swap_endianess_4((unsigned*)&arr[0]);
    swap_endianess_4((unsigned*)&arr[1]);
    swap_endianess_4((unsigned*)&arr[2]);
    
    tsm->add_triangle(arr[0], arr[1], arr[2]);
    //cout << arr[0] << ", " << arr[1] << ", " << arr[2] << endl;
  }
  return 0;
}

template <class FLD>
int
read_scalar_lookup(FLD *fld, int n, ifstream &str) {
  fld->resize_fdata();
  typedef typename FLD::value_type val_t;
  //val_t last = 0;
  int vset = 0;
  check_error(str);

  for(int i = 0; i < n; i++) {
    typedef typename FLD::value_type val_t;
    val_t val;
    str.read((char*)&val, sizeof(val_t));
    check_error(str);
    swap_endianess_4((unsigned*)&val);
    vset++;
    fld->set_value(val, (typename FLD::mesh_type::Face::index_type)i);    
  }
  return 0;
}


int
main(int argc, char **argv) {

  int status = 0;
  if (argc < 3 || argc > 5) {
    printUsageInfo(argv[0]);
    return 2;
  }

  SCIRunInit();
  setDefaults();

  TSMesh *tsm = scinew TSMesh();
  //exit(5);
  char *in = argv[1];
  char *out = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    status = 1;
    return 1;
  }
  ifstream vtk(in, ios::binary);
  check_error(vtk);

  char id[256], header[256], format[256];

  vtk.getline(id, 256);
  vtk.getline(header, 256);

  vtk >> format;
  cout << format << endl;  

  string dataset;
  vtk >> dataset;

  if (dataset != "DATASET") {
    cerr << "ERROR: expected DATASET keyword." << endl;
    status = 5;
    return 5;
  } else {    
      vtk >> dataset;
      if (dataset != "POLYDATA") {
	cerr << "ERROR: can only handle POLYDATA type vtk files at this time. "
	     << "got: " << dataset << endl;
	return 5;
      }
  }
  check_error(vtk);
  
  string attrib;
  vtk >> attrib;

  int n;
  vtk >> n;

  string type;
  vtk >> type;
  //cout << "attrib is : " << attrib << " " << n << " type is " << type << endl;
  check_error(vtk);
  vtk.get(); // eat a newline

  read_n_points<float>(tsm, n, vtk);
  
  check_error(vtk);

  string poly;
  vtk >> poly;

  if (poly != "POLYGONS") {
    cerr << "ERROR: can only handle POLYGONS type polygonal data " 
	 << " files at this time. got: " << poly << endl;
    return 1;
  }
  vtk >> n;
  int sz;
  vtk >> sz;
  cout << poly << " " << n << " " << sz << endl;

  vtk.get(); // eat a newline
  read_n_polys<unsigned>(tsm, n, vtk);
  check_error(vtk);
  
  string dat;
  vtk >> dat; 
  vtk >> n;

  check_error(vtk)
  cout << dat << " " << n << endl;

  if (dat == "CELL_DATA") {
    vtk >> dat; 
    vtk >> n;
    check_error(vtk)
    cout << dat << " " << n << endl;
  }
  
  

  FieldHandle ts_handle;
  string data, name;
//   vtk >> data;
//   check_error(vtk);
//   cout << data << endl;
//   vtk >> name;
//   check_error(vtk);
//   cout << name << endl;
//   vtk >> type;
//   check_error(vtk);
//   cout << type << endl;

  if (dat == "CELL_DATA") {
    if (type != "float") {
      cerr << "supporting float only atm..." << endl;
      return 1;
    }
    TSFieldC *tsc = scinew TSFieldC(TSMesh::handle_type(tsm));
    tsc->resize_fdata();
    //cout << "putting data at faces" << endl;
    string table, tname;
    vtk >> table >> tname;
    //cout << table << " " << tname << endl;
    vtk.get(); // eat a newline
    read_scalar_lookup(tsc, n, vtk);
    ts_handle = tsc;
  } else {
    // node centered data ...
    if (type != "float") {
      cerr << "supporting float only atm..." << endl;
      return 1;
    }
    TSFieldL *tsl = scinew TSFieldL(TSMesh::handle_type(tsm));
    tsl->resize_fdata();

    string table, tname;
    vtk >> table >> tname;
    //cout << table << " " << tname << endl;
    vtk.get(); // eat a newline
    read_scalar_lookup(tsl, n, vtk);
    ts_handle = tsl;
  }

  while (!vtk.eof()) {
    vtk.get();
  }



  if (bin_out) {
    BinaryPiostream out_stream(out, Piostream::Write);
    Pio(out_stream, ts_handle);
  } else {
    TextPiostream out_stream(out, Piostream::Write);
    Pio(out_stream, ts_handle);
  }

  return status;  
}    
