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

#include <Core/Geometry/Vector.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/HexTricubicHmtScaleFactorsEdges.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Init/init.h>
#include <StandAlone/convert/FileUtils.h>
#include <sci_hash_set.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace SCIRun;

typedef HexTricubicHmtScaleFactorsEdges<Vector>              FDVectorBasis;
typedef HexTricubicHmtScaleFactorsEdges<double>              FDdoubleBasis;
typedef NoDataBasis<double>                                  NDBasis;
typedef HexVolMesh<HexTricubicHmtScaleFactorsEdges<Point> >  HVMesh;
typedef GenericField<HVMesh, FDVectorBasis, vector<Vector> > HVFieldV;       
typedef GenericField<HVMesh, NDBasis, vector<double> >       HVField;   

using std::cerr;
using std::ifstream;
using std::endl;

struct edge_hsh {  
  int e1_;
  int e2_;
  double sf_;
  edge_hsh() {}
  edge_hsh(int e1, int e2, double sf) 
  {
    if (e1 < e2) {
      e1_ = e1;
      e2_ = e2;
    } else {
      e1_ = e2;
      e2_ = e1;
    }
    sf_ = sf;
  }
  size_t operator()(const edge_hsh &e) const
  {
    static int shift = sizeof(size_t) * 2;
    size_t rval = (e.e1_ << shift) | e.e2_;
    //    std::cerr << "shift: " << shift << " hash: " << rval
    //         << " e1: " << e.e1_ << " e2: " << e.e2_ << std::endl; 
    return rval;
  }

  bool operator()(const edge_hsh &e1, const edge_hsh &e2) const
  {
    return (e1.e1_ == e2.e1_ && e1.e2_ == e2.e2_);
  }
#if defined(__ECC) || defined(_MSC_VER)

    static const size_t bucket_size = 4;
    static const size_t min_buckets = 8;
#endif

};

#if defined(__ECC) || defined(_MSC_VER)
typedef hash_set<edge_hsh, edge_hsh> etable_t;
#else
typedef hash_set<edge_hsh, edge_hsh, edge_hsh> etable_t;
#endif

int
parse_edge_sf_17(ifstream &nodal_in, etable_t &tbl) 
{
  std::cerr << "in parse_edge_sf_17: " << std::endl;
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;
    double x[17];
    for (int j=0; j <= 17; j++)
    {
      nodal_in >> x[j];
    }
    edge_hsh e((int)x[5] - 1, (int)x[6] - 1, x[16]);
    tbl.insert(e);
  }
  return npts;
}


int
parse_ho_data33(ifstream &nodal_in, vector<Vector> &data_vals, HVMesh *hvm) 
{
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;
    double x[3], xdx[3], xdy[3], xdxy[3], xdz[3], xdyz[3], xdxz[3], xdxyz[3];
    for (int j=0; j<3; j++) 
      nodal_in >> x[j] >> xdx[j] >> xdy[j] >> xdxy[j] >> xdz[j] 
		>> xdyz[j] >> xdxz[j] >> xdxyz[j];

    if (nodal_in.eof()) break;

    double f, fdx, fdy, fdxy, fdz, fdyz, fdxz, fdxyz;
    nodal_in >> f >> fdx >> fdy >> fdxy >> fdz 
	     >> fdyz >> fdxz >> fdxyz;

    string label;
    int n;
    nodal_in >> label >> n;
    //    const double sf = 12.0;
    hvm->add_point(Point(x[0],x[1],x[2]));
    vector<Point> arr(7);
    arr[0].x(xdx[0]);
    arr[0].y(xdx[1]);
    arr[0].z(xdx[2]);
    arr[1].x(xdy[0]);
    arr[1].y(xdy[1]);
    arr[1].z(xdy[2]);
    arr[2].x(xdz[0]);
    arr[2].y(xdz[1]);
    arr[2].z(xdz[2]);
    arr[3].x(xdxy[0]);
    arr[3].y(xdxy[1]);
    arr[3].z(xdxy[2]);
    arr[4].x(xdyz[0]);
    arr[4].y(xdyz[1]);
    arr[4].z(xdyz[2]);
    arr[5].x(xdxz[0]);
    arr[5].y(xdxz[1]);
    arr[5].z(xdxz[2]);
    arr[6].x(xdxyz[0]);
    arr[6].y(xdxyz[1]);
    arr[6].z(xdxyz[2]);
    hvm->get_basis().add_derivatives(arr);

    cerr << "Added point #" << npts << ": (" << x[0] << ", " << x[1] 
	 << ", " << x[2] << ")" << endl;
  }

  cerr << "done adding points.\n";
  return npts;
}

int
parse_ho_data28(ifstream &nodal_in, vector<Vector> &data_vals, HVMesh *hvm) 
{
  int npts = 0;
  while (! nodal_in.eof()) {
    ++npts;
    double x[3], xdx[3], xdy[3], xdxy[3], xdz[3], xdyz[3], xdxz[3], xdxyz[3];
    for (int j=0; j<3; j++) 
      nodal_in >> x[j] >> xdx[j] >> xdy[j] >> xdxy[j] >> xdz[j] 
		>> xdyz[j] >> xdxz[j] >> xdxyz[j];

    if (nodal_in.eof()) break;

    double f[3];
    for (int j=0; j<3; j++) 
      nodal_in >> f[j];

    data_vals.push_back(Vector(f[0],f[1],f[2]));

    string label;
    int n;
    nodal_in >> label >> n;
    //    const double sf = 12.0;
    hvm->add_point(Point(x[0],x[1],x[2]));
    vector<Point> arr(7);
    arr[0].x(xdx[0]);
    arr[0].y(xdx[1]);
    arr[0].z(xdx[2]);
    arr[1].x(xdy[0]);
    arr[1].y(xdy[1]);
    arr[1].z(xdy[2]);
    arr[2].x(xdz[0]);
    arr[2].y(xdz[1]);
    arr[2].z(xdz[2]);
    arr[3].x(xdxy[0]);
    arr[3].y(xdxy[1]);
    arr[3].z(xdxy[2]);
    arr[4].x(xdyz[0]);
    arr[4].y(xdyz[1]);
    arr[4].z(xdyz[2]);
    arr[5].x(xdxz[0]);
    arr[5].y(xdxz[1]);
    arr[5].z(xdxz[2]);
    arr[6].x(xdxyz[0]);
    arr[6].y(xdxyz[1]);
    arr[6].z(xdxyz[2]);
    hvm->get_basis().add_derivatives(arr);

    cerr << "Added point #" << npts << ": (" << x[0] << ", " << x[1] 
	 << ", " << x[2] << ")" << endl;
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
    cerr << "4th is: " << argv[4] << endl;
    aux_file = argv[4];
  }

  // return 0;

  if (nodal_file == 0 || elem_file == 0 || out_file == 0) {
    cerr << "Error: expecting 3 arguments: <nodal data file> "
	 << "<element data file> <output file name>" << endl;
    return 1;
  }
  
  vector<string> headers;
  ifstream nodal_in(nodal_file);
  if (! nodal_in) {
    std::cerr << "Could not open file: " << nodal_file << std::endl;
    return 256;
  }

  //count the columns:
  char c;
  string hdr;
  do{
    nodal_in.get(c);
    if (c == '\t') { 
      headers.push_back(hdr);
      hdr.clear();
    } else {
      hdr += c;
    }
  } while (c != '\n');

  vector<string> aux_headers;
  ifstream aux_nodal_in(aux_file);

  if (! aux_nodal_in) {
    std::cerr << "Could not open file: " << aux_file << std::endl;
    return 256;
  }

  //count the columns:
  hdr.clear();
  do{
    aux_nodal_in.get(c);
    if (c == '\t') { 
      aux_headers.push_back(hdr);
      hdr.clear();
    } else {
      hdr += c;
    }
  } while (c != '\n');

  if (headers.size()) {
    cout << "Parsing labels:" << endl;
    vector<string>::iterator hiter = headers.begin();
    while (hiter != headers.end()) {
      cout << *hiter++ << ", " << endl;
    }
    cout << endl << "There are " << headers.size() << " columns." << endl;
  }

  if (aux_headers.size()) {
    cout << "Parsing labels:" << endl;
    vector<string>::iterator hiter = aux_headers.begin();
    while (hiter != aux_headers.end()) {
      cout << *hiter++ << ", " << endl;
    }
    cout << endl << "AUX: There are " << aux_headers.size() << " columns." << endl;
  }

  int npts = 0;
  vector<Vector> data_vals;
  vector<double> data_vals_scalar;


  if (headers.size() == 33) {
    npts = parse_ho_data33(nodal_in, data_vals, hvm);
  } else if (headers.size() == 28) {
    npts = parse_ho_data28(nodal_in, data_vals, hvm);
  } else {
    cerr << "Dont know what to do with " << headers.size()
	 << " columns of data" << endl;
    return (1 << 1);
  }
  
  etable_t tbl;
  if (aux_file) {
    if (aux_headers.size() == 17) {
      npts = parse_edge_sf_17(aux_nodal_in, tbl);
    } else {
      cerr << "AUX: Dont know what to do with " << aux_headers.size()
	   << " columns of aux data" << endl;
      return (1 << 1);
    }
  }

  int nhexes = getNumNonEmptyLines(elem_file) - 1;
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


  std::cerr << "tbl size: " << tbl.size() << std::endl;
  if (tbl.size() > 0) {
    std::cerr << "looking up edges" << std::endl;
    hvm->synchronize(Mesh::EDGES_E);
    HVMesh::Edge::iterator iter, end;
    hvm->begin(iter);
    hvm->end(end);
    int ecount = 0;
    double max = 0.0;
    while (iter != end) {
      HVMesh::Node::array_type arr;
      hvm->get_nodes(arr, *iter);
      ++iter;
      edge_hsh e(arr[0], arr[1], 0.0L);
      etable_t::iterator eptr = tbl.find(e);
      if (eptr == tbl.end()) {
	std::cerr << "ERROR looking up edge" << std::endl;
	return 69;
      }
      e = *eptr;
      std::cerr << "adding sf: " << e.sf_ << std::endl;
      vector<double> sf;
      sf.push_back(e.sf_);
      if (e.sf_ > max) max = e.sf_;
      hvm->get_basis().add_scalefactors(sf);
      ecount++;
    }
    std::cerr << "added "  << ecount << " scale factors. (max:" 
	      << max << ")" << std::endl;
  }

  FieldHandle hvH;  
  if (0) { 
    cerr << "loading in strain data" << endl;
    HVField *hv = scinew HVField(hvm);
    hv->resize_fdata();
    hv->fdata() = data_vals_scalar;
    hvH = hv;
  } else if (headers.size() == 33 || headers.size() == 28) {
    cerr << "loading in no data" << endl;
    HVField *hv = scinew HVField(hvm);
    hv->resize_fdata();
    //hv->fdata() = data_vals;
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
