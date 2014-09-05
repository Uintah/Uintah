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

#include <Core/Datatypes/Field.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/LatVolField.h>
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
vector<unsigned> fcount;
vector<unsigned> num_files;
vector<string> prename;

const unsigned int max_lines = 25000000;

inline double
round(double var)
{
  return ((var)>=0?(int)((var)+0.5):(int)((var)-0.5));
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
      fname << ios::right << 1;
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

  int sizex = (int)(round(maxb.x() - minb.x()) / max_vol_s);
  int sizey = (int)(round(maxb.y() - minb.y()) / max_vol_s);
  int sizez = (int)(round(maxb.z() - minb.z()) / max_vol_s);
  
  LatVolMesh *cntrl = new LatVolMesh(sizex, sizey, sizez, minb, maxb);
  double vol = cntrl->get_volume(LatVolMesh::Cell::index_type(cntrl,0,0,0));
  LatVolMesh::Cell::iterator iter, end;
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
	fname << ios::right << ++num_files[val];
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
  if (argc != 3)
  {
    cout << "Usage:  GenerateMPMData <input-text-field> <dest-dir>" << endl;
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

  string outdir(argv[2]);

  FieldHandle field_handle;
  
  Piostream *in_stream = auto_istream(argv[1]);
  if (!in_stream)
  {
    cout << "Error reading file " << argv[1] << ".\n";
    exit(101);
  }

  Pio(*in_stream, field_handle);
  delete in_stream;

  
  write_MPM<TetVolField<int> >(field_handle, outdir);


  return 0;
}    
