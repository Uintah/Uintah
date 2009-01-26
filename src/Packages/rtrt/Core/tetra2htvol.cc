/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <unistd.h>
#include <cstdio>

using namespace std;

void usage(char *progname)
{
  cout << "Usage: " << progname << " tetra_file pts_file data_file output_file\n";
  exit(1);
}

int main(int argc, char *argv[])
{
  if (argc != 5) usage(argv[0]);

  char *tet_file, *pts_file, *data_file, *out_file;

  tet_file = argv[1];
  pts_file = argv[2];
  data_file = argv[3];
  out_file = argv[4];

  ifstream in_tet(tet_file);
  if(!in_tet){
    cerr << "Error opening input file: " << tet_file << '\n';
    exit(1);
  }
  ifstream in_pts(pts_file);
  if(!in_pts){
    cerr << "Error opening input file: " << pts_file << '\n';
    exit(1);
  }
  ifstream in_data(data_file);
  if(!in_data){
    cerr << "Error opening input file: " << data_file << '\n';
    exit(1);
  }

  int ntetra, npts, ndata;
  in_tet >> ntetra;
  in_pts >> npts;
  in_data >> ndata;

  if(npts != ndata) {
    cerr << "Error, number of data points (" << npts << ") does not\n";
    cerr << "match number of data elements (" << ndata << ").\n";
    exit(1);
  }

  int i;
  float *points=new float[4*npts];
  for(i=0; i < npts; i++) {
    in_pts >> points[4*i] >> points[4*i+1] >> points[4*i+2];
    in_data >> points[4*i+3];
  }
  int *tetra=new int[4*ntetra];
  for(i=0; i < ntetra; i++) {
    int foo;
    in_tet >> tetra[4*i] >> tetra[4*i+1] >> tetra[4*i+2] >> tetra[4*i+3] >> foo;
    tetra[4*i]--;
    tetra[4*i+1]--;
    tetra[4*i+2]--;
    tetra[4*i+3]--;
  }

  if(!in_tet){
    cerr << "Error reading tetra file: " << tet_file << '\n';
    exit(1);
  }
  if(!in_pts){
    cerr << "Error reading points file: " << pts_file << '\n';
    exit(1);
  }
  if(!in_data){
    cerr << "Error reading data file: " << data_file << '\n';
    exit(1);
  }

  ofstream out_htvolfile(out_file);
  if(!out_htvolfile){
    cerr << "Error HTVolumeBrick output file for outputfile: " << out_file << '\n';
    exit(1);
  }

  //  ASCII header atop all HTVolumeBrick files
  out_htvolfile << "HTVolumeBrick file\n";
  out_htvolfile << npts << " " << ntetra << "\n";

  out_htvolfile.write((char *) points, (int) sizeof(float) * 4 * npts);
  out_htvolfile.write((char *) tetra, (int) sizeof(float) * 4 * ntetra);

}
