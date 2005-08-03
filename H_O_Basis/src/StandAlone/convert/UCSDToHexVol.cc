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

//    File   : TextToHexTricubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004


// This program will read in a UCSD style .xls (specifying node variables of each 
// point, one per line, entries separated by white space, file can have 
// an optional one line header specifying number of points... and if it
// doesn't, you have to use the -noPtsCount command-line argument) and a
// .hex file (specifying i/j/k/l/m/n/o/p indices for each hex, also one per 
// line, again with an optional one line header (use -noElementsCount if it's 
// not there).  The hex entries are assumed to be zero-based, unless you 
// specify -oneBasedIndexing.  And the SCIRun output file is written in 
// ASCII, unless you specify -binOutput.

#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <Core/Init/init.h>
#include <StandAlone/convert/FileUtils.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace SCIRun;

typedef HexTrilinearLgn<Vector>                FDVectorBasis;
typedef HexTrilinearLgn<double>                FDdoubleBasis;
typedef HexVolMesh<HexTrilinearLgn<Point> >    HVMesh;
typedef GenericField<HVMesh, FDVectorBasis, vector<Vector> > HVFieldV;       
typedef GenericField<HVMesh, FDdoubleBasis, vector<double> > HVField;   

using std::cerr;
using std::ifstream;
using std::endl;

int
parse_lin_vec_data(ifstream &nodal_in, vector<Vector> &data_vals, 
		   HVMesh *hvm) 
{
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;
    double x[3];
    for (int j=0; j<3; j++) 
      nodal_in >> x[j];

    if (nodal_in.eof()) break;

    double f[3];
    for (int j=0; j<3; j++) 
      nodal_in >> f[j];

    Vector fib_dir(f[0], f[1], f[2]);
    data_vals.push_back(fib_dir);

    string label;
    int n;
    nodal_in >> label >> n;
    hvm->add_point(Point(x[0],x[1],x[2]));
  }
  cerr << "done adding " << npts - 1 << " points." << endl;
  return npts;
}


int
parse_lin_strain_data25(ifstream &nodal_in, vector<double> &data_vals, 
			HVMesh *hvm) 
{
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;

    double x[24];
    for (int j=0; j<24; j++) 
      nodal_in >> x[j];

    if (nodal_in.eof()) break;
    
    int node_index;
    nodal_in >> node_index;
    //make a scalar field out of this.

    data_vals.push_back(x[0]);
  }
  cerr << "done adding " << npts - 1 << " points." << endl;
  return npts;
}

int
parse_lin_strain_data4(ifstream &nodal_in, vector<double> &data_vals, 
		      HVMesh *hvm) 
{
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;

    double x[3];
    for (int j=0; j<3; j++) 
      nodal_in >> x[j];

    if (nodal_in.eof()) break;
    // 2nd order tensor
    double eff = 0.0;
    
    int node_index;
    nodal_in >> node_index;
    //make a scalar field out of this.

    data_vals.push_back(eff);
    hvm->add_point(Point(x[0],x[1],x[2]));
  }
  cerr << "done adding " << npts - 1 << " points." << endl;
  return npts;
}

int
parse_lin_strain_data5(ifstream &nodal_in, vector<double> &data_vals, 
		      HVMesh *hvm, bool data = true) 
{
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;

    double x[3];
    for (int j=0; j<3; j++) 
      nodal_in >> x[j];

    if (nodal_in.eof()) break;
    // 2nd order tensor
    double eff;
    string lab;
    if (data) {
      nodal_in >> eff;
    } else {
      nodal_in >> lab;
      eff = 0.0;
    }
    int node_index;
    nodal_in >> node_index;
    //make a scalar field out of this.

    data_vals.push_back(eff);
    hvm->add_point(Point(x[0],x[1],x[2]));
  }
  cerr << "done adding " << npts - 1 << " points." << endl;
  return npts;
}

int
parse_lin_strain_data(ifstream &nodal_in, vector<double> &data_vals, 
		      HVMesh *hvm) 
{
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;
    int node_index;
    nodal_in >> node_index;
    double x[3];
    for (int j=0; j<3; j++) 
      nodal_in >> x[j];

    if (nodal_in.eof()) break;
    // 2nd order tensor
    double t[6];
    for (int j=0; j<6; j++) 
      nodal_in >> t[j];
    
    // cartesian diagonal of the tensor
    double diag[3];
    for (int j=0; j<3; j++) 
      nodal_in >> diag[j];

    //make a scalar field out of this for now.

    data_vals.push_back(t[0]);
    hvm->add_point(Point(x[0],x[1],x[2]));
  }
  cerr << "done adding " << npts - 1 << " points." << endl;
  return npts;
}

int
parse_ho_data(ifstream &nodal_in, vector<Vector> &data_vals, HVMesh *hvm) 
{
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;
    double x[3], xdx[3], xdy[3], xdxy[3], xdz[3], xdyz[3], xdxz[3], xdxyz[3];
    for (int j=0; j<3; j++) 
      nodal_in >> x[j] >> xdx[j] >> xdy[j] >> xdxy[j] >> xdz[j] 
		>> xdyz[j] >> xdxz[j] >> xdxyz[j];

    if (nodal_in.eof()) break;

    double f[3], fdx[3], fdy[3], fdxy[3], fdz[3], fdyz[3], fdxz[3], fdxyz[3];
    for (int j=0; j<3; j++) 
      nodal_in >> f[j] >> fdx[j] >> fdy[j] >> fdxy[j] >> fdz[j] 
		>> fdyz[j] >> fdxz[j] >> fdxyz[j];

    double phi, theta;
    const double torad = 3.145926535898 / 180.;
    // convert to radians
    phi = f[0] * torad;
    theta = f[1] * torad;
    // create unit vectors
    Vector fib_dir;
    fib_dir.x(sin(phi) * cos(theta));
    fib_dir.y(sin(phi) * sin(theta));
    fib_dir.z(cos(phi));

    //cerr << fib_dir << " len ->" << fib_dir.length() << endl;
    data_vals.push_back(fib_dir);

    double v[3][3], vdx[3][3], vdy[3][3], vdxy[3][3], vdz[3][3], 
      vdyz[3][3], vdxz[3][3], vdxyz[3][3];
    for (int j=0; j<3; j++) 
      for (int k=0; k<3; k++) 
	nodal_in >> v[j][k] >> vdx[j][k] >> vdy[j][k] >> vdxy[j][k] 
		  >> vdz[j][k] >> vdyz[j][k] >> vdxz[j][k] >> vdxyz[j][k];

    string label;
    int n;
    nodal_in >> label >> n;
    //    const double sf = 12.0;
    hvm->add_point(Point(x[0],x[1],x[2]));
  }

  cerr << "done adding points.\n";
  return npts;
}


int
main(int argc, char **argv) {

  SCIRunInit();
  HVMesh *hvm = new HVMesh();


  
  char *nodal_file = argv[1];
  char *elem_file = argv[2];
  char *out_file = argv[3];
  char *aux_file = 0;
  if (argc == 5) {
    cerr << "5th is: " << argv[4] << endl;
    aux_file = argv[4];
  }

  // return 0;

  if (nodal_file == 0 || elem_file == 0 || out_file == 0) {
    cerr << "Error: expecting 3 arguments: <nodal data file> "
	 << "<element data file> <output file name>" << endl;
    return 1;
  }
  
  ifstream col_str(nodal_file);
  //count the columns:
  int cols = 1;
  char c;
  do{
    col_str.get(c);
    if (c == '\t') { 
      cols++; 
    }
  } while (c != '\n');
  col_str.close();


  ifstream aux_col_str(aux_file);
  int aux_cols = 1;
  if (aux_file) {
    char c;
    do{
      aux_col_str.get(c);
      if (c == '\t') { 
	aux_cols++; 
      }
    } while (c != '\n');
    aux_col_str.close();
  }

  ifstream nodal_in(nodal_file);
  ifstream aux_nodal_in(aux_file);

  bool header = true;
  vector<string> labels(cols);
  if (header) {
    cout << "Parsing labels:" << endl;
    for (int i = 0; i < cols; ++i) {
      string str;
      nodal_in >> str;
      cout << str << ", " << endl;
      labels[i] = str;
    }
    cout << endl << "There are " << labels.size() << " columns." << endl;
  }

  if (header && aux_file) {
    vector<string> labels(aux_cols);
    cout << "Parsing aux labels:" << endl;
    for (int i = 0; i < aux_cols; ++i) {
      string str;
      aux_nodal_in >> str;
      cout << str << ", " << endl;
      labels[i] = str;
    }
    cout << endl << "There are " << labels.size() << " aux columns." << endl;
  }

  int npts = 0;
  vector<Vector> data_vals;
  vector<double> data_vals_scalar;

  int use_cols = 0;
  if (aux_file) {
    use_cols = aux_cols;
  } else {
    use_cols = cols;
  }

  if (cols == 8) {
    npts = parse_lin_vec_data(nodal_in, data_vals, hvm);
  } else if (cols == 122) {
    npts = parse_ho_data(nodal_in, data_vals, hvm);
  } else if (cols == 25) {
    npts = parse_lin_strain_data25(nodal_in, data_vals_scalar, hvm);
  } else if (cols == 13) {
    npts = parse_lin_strain_data(nodal_in, data_vals_scalar, hvm);
  } else if (cols == 5) {
    if (labels[3] == string("Label")) {
      npts = parse_lin_strain_data5(nodal_in, data_vals_scalar, hvm, false);
    } else {
      npts = parse_lin_strain_data5(nodal_in, data_vals_scalar, hvm);
    }
  } else if (cols == 4) {
    npts = parse_lin_strain_data4(nodal_in, data_vals_scalar, hvm);
  } else {
    cerr << "Dont know what to do with " << cols 
	 << " columns of data" << endl;
    return (1 << 1);
  }
  
  if (aux_file) {
    vector<Vector> aux_data_vals;
    vector<double> aux_data_vals_scalar;
    if (aux_cols == 8) {
      npts = parse_lin_vec_data(aux_nodal_in, aux_data_vals, hvm);
    } else if (aux_cols == 122) {
      npts = parse_ho_data(aux_nodal_in, aux_data_vals, hvm);
    } else if (aux_cols == 25) {
      npts = parse_lin_strain_data25(aux_nodal_in, aux_data_vals_scalar, hvm);
    } else if (aux_cols == 13) {
      npts = parse_lin_strain_data(aux_nodal_in, aux_data_vals_scalar, hvm);
    } else if (aux_cols == 5) {
      npts = parse_lin_strain_data5(aux_nodal_in, aux_data_vals_scalar, hvm);
    } else {
      cerr << "Dont know what to do with " << aux_cols 
	   << " columns of data" << endl;
      return (1 << 1);
    }
  }

  int nhexes;
  nhexes = getNumNonEmptyLines(elem_file) - 1;
  ifstream hexesstream(elem_file);

  cerr << "number of hexes = "<< nhexes <<"\n";

  // parse header out.
  vector<string> eheader;
  for (int i = 0; i < 10; i++) {
    string s;
    hexesstream >> s;
    eheader.push_back(s);
  }
  vector<string>::iterator iter = eheader.begin();
  while (iter != eheader.end()) {
    cerr << *iter++ << endl;
  }
  int baseIndex = 1;
  for (int i = 0; i < nhexes; i++) {
    int n1, n2, n3, n4, n5, n6, n7, n8;
    string label;
    int e;
    hexesstream >> n1 >> n2 >> n3 >> n4 >> n5 >> n6 >> n7 >> n8 >> label >> e;
    n1-=baseIndex; 
    n2-=baseIndex; 
    n3-=baseIndex; 
    n4-=baseIndex; 
    n5-=baseIndex; 
    n6-=baseIndex; 
    n7-=baseIndex; 
    n8-=baseIndex; 
    if (n1<0 || n1>=npts) { 
      cerr << "Error -- n1 ("<<i<<") out of bounds: "<<n1<<"\n"; 
      return 0; 
    }
    if (n2<0 || n2>=npts) { 
      cerr << "Error -- n2 ("<<i<<") out of bounds: "<<n2<<"\n"; 
      return 0; 
    }
    if (n3<0 || n3>=npts) { 
      cerr << "Error -- n3 ("<<i<<") out of bounds: "<<n3<<"\n"; 
      return 0; 
    }
    if (n4<0 || n4>=npts) { 
      cerr << "Error -- n4 ("<<i<<") out of bounds: "<<n4<<"\n"; 
      return 0; 
    }
    if (n5<0 || n5>=npts) { 
      cerr << "Error -- n5 ("<<i<<") out of bounds: "<<n5<<"\n"; 
      return 0; 
    }
    if (n6<0 || n6>=npts) { 
      cerr << "Error -- n6 ("<<i<<") out of bounds: "<<n6<<"\n"; 
      return 0; 
    }
    if (n7<0 || n7>=npts) { 
      cerr << "Error -- n7 ("<<i<<") out of bounds: "<<n7<<"\n"; 
      return 0; 
    }
    if (n8<0 || n8>=npts) { 
      cerr << "Error -- n8 ("<<i<<") out of bounds: "<<n8<<"\n"; 
      return 0; 
    }
    
    hvm->add_hex(n1, n2, n4, n3, n5, n6, n8, n7);
//     cerr << "Added hex #"<< i << ": [" << n1 << " " << n2 << " " << n3 << " "
// 	 << n4 << " " << n5 << " " << n6 << " " << n7 << " " << n8 << "]" 
// 	 << endl;
  }
  cerr << "done adding elements.\n";

  FieldHandle hvH;  
  if (cols == 13 || cols == 5 || cols == 25) { 
    cerr << "loading in strain data" << endl;
    HVField *hv = scinew HVField(hvm);
    hv->resize_fdata();
    hv->fdata() = data_vals_scalar;
    hvH = hv;
  } else {
    cerr << "loading in vector data" << endl;
    HVFieldV *hv = scinew HVFieldV(hvm);
    hv->resize_fdata();
    hv->fdata() = data_vals;
    hvH = hv;
  }

  BinaryPiostream out_stream(out_file, Piostream::Write);
  Pio(out_stream, hvH);
  return 0;  
}    
