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
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <sys/stat.h>
#include <map>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

vector<ofstream*> *files = 0;

template<class Fld>
void
write_MPM(FieldHandle& field_h, const string &outdir)
{
  Fld *ifield = (Fld *) field_h.get_rep();
  bool have_files = false;
  vector<pair<string, Tensor> > field_tensors;
  if (ifield->get_property("conductivity_table", field_tensors)) {
    have_files = true;
    files = new vector<ofstream*>;
    vector<pair<string, Tensor> >::iterator titer = field_tensors.begin();
    while (titer != field_tensors.end()) {
      files->push_back(new ofstream((outdir + string("/") + 
				     titer++->first).c_str(), ios_base::out));
    }    
  }

  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
    
  typename Fld::mesh_type::Cell::iterator iter, end;
  mesh->begin( iter );
  mesh->end( end );
  while (iter != end) {
    double vol = mesh->get_volume(*iter);
    Point c;
    mesh->get_center(c, *iter);
    int val = ifield->value(*iter);
    ++iter;
    ofstream* str = (*files)[val];
    (*str) << setprecision (9) <<c.x() << " " << c.y() << " " << c.z() 
	   << " " << vol << endl;
  }
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
  if (stat(argv[1], &buff))
  {
    cout << "File " << argv[1] << " not found\n";
    exit(99);
  }

  if (stat(argv[2], &buff))
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
