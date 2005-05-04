//***********************************************************************
//  Cauchy geometry format (.geo) 
//      converting to 
//  SCIRun compatible node(.pts) & mesh(.tet) format
//***********************************************************************

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int
main(int argc, char* argv[])
{

  if(argc !=4) {
    cout << "geoTopts <element type> INPUT OUTPUT " << endl;
    cout << "   INPUT  : input filename having .geo extension" << endl;
    cout << "   OUTPUT : output filename of SCIRun readable node format(.pts)" << endl;
    cout << "            and mesh format (.fac .tet .hex) depending on mesh type" << endl;
    cout << "           (filename extensions(.pts & .fac .tet .hex) are automatically attached to OUTPUT)" << endl;
    cout << "   -s     :  triangular surface element" << endl;
    cout << "   -t     :  tetrahedra volume element" << endl;
    cout << "   -h     :  hexagonal volume element" << endl;    
    exit(1);
  }

  ifstream geoFile(argv[2], ios::in);
  
  if (!geoFile){
    cerr << "File could not be opened" << endl;
    exit(1);
  }

  string tmp;

  // skipping text lines
  for (int i=0; i<4; ++i){
    getline(geoFile,tmp);
  }

  char c;
  int node, mesh,elem=0;
  double x,y,z;

  geoFile >> tmp >> tmp >> tmp>> c >> node;
  cout << "Number of Node = " << node << endl;

  geoFile >> tmp >> tmp >> tmp >> c >> mesh;
  cout << "Number of Mesh = " << mesh << endl;

  // skipping text lines
  for (int i=0; i<6; ++i){
    getline(geoFile, tmp);
    //  cout << tmp << endl << endl;
  }

  const char *fname=0;
  string ext=".pts";
  string CompleteName=argv[3];
  CompleteName= argv[3]+ext;
  fname=CompleteName.c_str();

  ofstream ptsFile(fname,ios::out);
        
  // writing number of node
  ptsFile << node << endl;

  //writing node location
  for (int i=0; i<node; ++i){
    geoFile >> x >> y >> z;
    ptsFile << x << " " << y << " " << z << endl;
  }

  ptsFile.close();
        
  // skipping text lines
  for (int i=0; i<5; ++i){
    getline(geoFile, tmp);
    // cout << tmp << endl << endl;
  }
        
  if (strcmp(argv[1],"-t")==0){
    ext=".tet";
    elem=4;
    cout << " !! Tetrahedra element !! " << endl;
  }
  else if (strcmp(argv[1],"-s")==0){
    ext=".fac";
    elem=3;
    cout << " !! Triangular element !! " << endl;
  }
  else if (strcmp(argv[1],"-h")==0){
    ext=".hex";
    elem=8;
    cout << " !! Hexagon element !! " << endl;
  }


  CompleteName=argv[3];
  CompleteName= argv[3]+ext;
  fname=CompleteName.c_str();

  ofstream tetFile(fname,ios::out);
  tetFile << mesh << endl;
  cout << "reading connectivity:   " << flush;

  for (int i=0; i<mesh; ++i){
    if(!((i*20)%mesh))  cout << "+" << flush;

    getline(geoFile, tmp);

    int l=0, p;
    for (int j=0; j<elem; ++j){
      p=0;
      for(int k=0; k<6; ++k,++l){
        c=tmp[7+l];  // after skipping 7 characters in the beginning of each line
        if(c !=' ' ){
          p *=10;
          p +=atoi(&c);
        }
      }
      tetFile << p-1 << " ";  // node number begins from 0.
    }
    tetFile << endl;
  }

  cout << endl;
        
  geoFile.close();
  tetFile.close();
        
  return 0;
        
}
