#include <stdio.h>
#include <fstream>
#include <unistd.h>
#include <iostream>

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
