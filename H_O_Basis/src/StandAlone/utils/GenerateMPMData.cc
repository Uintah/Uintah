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
//    File   : GenerateMPMData.cc
//    Author : Martin Cole
//    Date   : Wed Jun  2 16:08:19 2004

#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Geometry/BBox.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <sys/stat.h>
#include <map>
#include <sstream>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

vector<ofstream*> *files = 0;
vector<ofstream*> *aux_files = 0;
vector<unsigned> fcount;
vector<unsigned> num_files;
vector<string> prename;

const unsigned int max_lines = 25000000;
double part_per_mm = 4.;

template<class Fld>
void
write_MPM_fibdir(FieldHandle& field_h, const string &outdir)
{
  // this outputs only one material
  Fld *ifield = (Fld *) field_h.get_rep();
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();

  files = new vector<ofstream*>;
  aux_files = new vector<ofstream*>;
  fcount.push_back(0);
  num_files.push_back(1);
  stringstream fname(stringstream::in | stringstream::out);
  string nm;
  fname << outdir << "/" << "heart-diastole";
  fname >> nm;
  prename.push_back(nm);

  fname.clear();
  fname << nm << "-"; 
  fname.fill('0');
  fname.width(4);
  fname << right << 1;
  string nm1;
  fname >> nm1;
  string aux_nm = nm1 + ".elems";
  files->push_back(new ofstream(nm1.c_str(), ios_base::out));
  aux_files->push_back(new ofstream(aux_nm.c_str(), ios_base::out));

#if defined(__sgi)
#  define round(var) ((var)>=0?(int)((var)+0.5):(int)((var)-0.5))
#endif

  mesh->synchronize(Mesh::NODES_E | Mesh::EDGES_E | 
		    Mesh::FACES_E | Mesh::CELLS_E);
  typename Fld::mesh_type::Cell::iterator iter, end;
  mesh->begin( iter );
  mesh->end( end );
  
  while (iter != end) {
    Point o, s, t, u;
    vector<double> coord(3, 0.0);
    mesh->interpolate(o, coord, *iter);
    coord[0] = 1.0;
    mesh->interpolate(s, coord, *iter);
    coord[0] = 0.0;
    coord[1] = 1.0;
    mesh->interpolate(t, coord, *iter);
    coord[1] = 0.0;
    coord[2] = 1.0;
    mesh->interpolate(u, coord, *iter);
    BBox elem_bbox;
    elem_bbox.extend(o);
    elem_bbox.extend(s);
    elem_bbox.extend(t);
    elem_bbox.extend(u);


    // assume length in mm's
    double dx = (s - o).length();
    double dy = (t - o).length();
    double dz = (u - o).length();

    double vol = dx * dy * dz; 
    
    const int div_per_x = (int)round(part_per_mm * dx);
    const int div_per_y = (int)round(part_per_mm * dy);
    const int div_per_z = (int)round(part_per_mm * dz);

    const double part_vol = vol / (div_per_x * div_per_y * div_per_z);
    
    dx = 1. / (double)div_per_x;
    dy = 1. / (double)div_per_y;
    dz = 1. / (double)div_per_z;

    for (double  k = dz * 0.5; k < 1.0; k += dz) {
      for (double j = dy * 0.5; j < 1.0; j += dy) {
	for (double i = dx * 0.5; i < 1.0; i += dx) {
	  
	  coord[0] = i;
	  coord[1] = j;
	  coord[2] = k;
	  //cerr << "at : [" << i << "," << j << "," << k << "]" << endl;
	  Point c;
	  mesh->interpolate(c, coord, *iter);
	  //sanity check..
// 	  if (! elem_bbox.inside(c)) {
// 	    cerr << "Error: interpolated to a point outside the element!" 
// 		 << endl;
// 	  }

	  Vector v;
	  ifield->interpolate(v, coord, *iter);

	  if (fcount[0] > max_lines) {
	    string nm;
	    stringstream fname(stringstream::in | stringstream::out);
	    fname << prename[0] << "-";
	    fname.fill('0');
	    fname.width(4);
	    fname << right << ++num_files[0];
	    fname >> nm;
	    string aux_nm = nm + ".elems";
	    delete (*files)[0];
	    (*files)[0] = new ofstream(nm.c_str(), ios_base::out);
	    (*aux_files)[0] = new ofstream(aux_nm.c_str(), ios_base::out);
	    fcount[0] = 0;
	  }

	  ofstream* str = (*files)[0];
	  ofstream* aux_str = (*aux_files)[0];
	  (*str) << setprecision (9) 
		 << c.x() << " " << c.y() << " " << c.z() << " " 
		 << part_vol << " "
		 << v.x() << " " << v.y() << " " << v.z() << endl;

	  // aux file just writes the elem number that the particle belongs to.
	  (*aux_str) << *iter << endl;
	  fcount[0]++;
	}
      }
    }
    cout << "." << *iter << ".";
    cout.flush();
    ++iter;
  }
  cout << endl;
}

template<class Fld>
void
write_MPM(FieldHandle& field_h, const string &outdir)
{

  Fld *ifield = (Fld *) field_h.get_rep();
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
  bool have_files = false;
  vector<pair<string, Tensor> > field_tensors;
  if (ifield->get_property("conductivity_table", field_tensors)) {
    have_files = true;
    files = new vector<ofstream*>;
    vector<pair<string, Tensor> >::iterator titer = field_tensors.begin();
    while (titer != field_tensors.end()) {
      fcount.push_back(0);
      num_files.push_back(1);
      stringstream fname(stringstream::in | stringstream::out);
      string nm;
      fname << outdir << "/" << titer++->first;
      fname >> nm;
      prename.push_back(nm);

      fname.clear();
      fname << nm << "-"; 
      fname.fill('0');
      fname.width(4);
      fname << right << 1;
      string nm1;
      fname >> nm1;
      files->push_back(new ofstream(nm1.c_str(), ios::out));

    }    
  }

  const double max_vol_s = 1.0L;
  BBox bbox = field_h->mesh()->get_bounding_box();
  Point minb, maxb;
  minb = bbox.min();
  maxb = bbox.max();

#if defined(__sgi)
#  define round(var) ((var)>=0?(int)((var)+0.5):(int)((var)-0.5))
#endif

  int sizex = (int)(round(maxb.x() - minb.x()) / max_vol_s);
  int sizey = (int)(round(maxb.y() - minb.y()) / max_vol_s);
  int sizez = (int)(round(maxb.z() - minb.z()) / max_vol_s);
  
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;

  LVMesh *cntrl = new LVMesh(sizex, sizey, sizez, minb, maxb);
  double vol = cntrl->get_volume(LVMesh::Cell::index_type(cntrl,0,0,0));
  LVMesh::Cell::iterator iter, end;
  cntrl->begin( iter );
  cntrl->end( end );
  int count = 0;
  
  while (iter != end) {
    Point c;
    cntrl->get_center(c, *iter);
    ++iter; count++;
    typename Fld::mesh_type::Cell::index_type ci;
    if (mesh->locate(ci, c)) {
      int val = ifield->value(ci);
      if (fcount[val] > max_lines) {
	string nm;
	stringstream fname(stringstream::in | stringstream::out);
	fname << prename[val] << "-";
	fname.fill('0');
	fname.width(4);
	fname << right << ++num_files[val];
	fname >> nm;
	delete (*files)[val];
	(*files)[val] = new ofstream(nm.c_str(), ios::out);
	fcount[val] = 0;
      }

      ofstream* str = (*files)[val];
      (*str) << setprecision (9) <<c.x() << " " << c.y() << " " << c.z() 
	     << " " << vol << endl;
       fcount[val]++;
    }
    
    if (count % ((int)(sizex*sizey*sizez*0.01)) == 0) { cout << "."; }
  }    
  cout << endl;
}





int
main(int argc, char **argv)
{
  if (argc != 4)
  {
    cout << "Usage:  GenerateMPMData <input-field> <dest-dir> <num particles per millimeter>" << endl;
    exit(0);
  }

  struct stat buff;
  if (stat(argv[1], &buff) == -1)
  {
    cout << "File " << argv[1] << " not found\n";
    exit(99);
  }

  if (stat(argv[2], &buff) == -1)
  {
    cout << "Directory  " << argv[2] << " not found\n";
    exit(100);
  }
  string ppm(argv[3]);
  part_per_mm = atof(ppm.c_str());


  string outdir(argv[2]);

  FieldHandle field_handle;
  
  Piostream *in_stream = auto_istream(argv[1]);
  if (!in_stream)
  {
    cout << "Error reading file " << argv[1] << ".\n";
    exit(101);
  }

  Pio(*in_stream, field_handle);
//   delete in_stream;
//   typedef TetVolMesh<TetLinearLgn<Point> > TVMesh;
//   typedef TetLinearLgn<int>  DatBasis;
//   typedef GenericField<TVMesh, DatBasis,    vector<int> > TVField;
//   write_MPM<TVField>(field_handle, outdir);

  typedef HexVolMesh<HexTrilinearLgn<Point> >    HVLinMesh;
  typedef HexTrilinearLgn<double>                ScaDatBasis;
  typedef HexTrilinearLgn<Vector>                VecDatBasis;
  typedef GenericField<HVLinMesh, ScaDatBasis, vector<double> > HVLinSField;
  typedef GenericField<HVLinMesh, VecDatBasis, vector<Vector> > HVLinVField;

  write_MPM_fibdir<HVLinVField>(field_handle, outdir);

  return 0;
}    
