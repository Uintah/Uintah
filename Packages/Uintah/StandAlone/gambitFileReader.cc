// To compile:  g++ -o gambitFileReader gambitFileReader.cc
// This will take in a data file from GAMBIT
//////////////////////////////////////////////////

#include <iostream>                    
#include <string>                      
#include <fstream>                     
#include <math.h>
#define d_SMALL_NUM 1e-100;
using namespace std;

int x, y;

int main()
{
  ifstream instream;       
  ofstream pts_stream,tri_stream;
  
  int numNodes = 0;
  int numElements = 0;
  string name,out_name, info, info1;
  cout << endl << "**********************************************" << endl;
  cout << "Input the name of the Gambit output file: ";
  cin >> name;

  cout << "Input the output file name for storing points and surfaces: ";
  cin >> out_name;
 
  instream.open(name.c_str());                       // opens files name
 
  string pts_name = out_name + ".pts";
  string tri_name = out_name + ".tri";
  pts_stream.open(pts_name.c_str());
  tri_stream.open(tri_name.c_str());
 
  while (!instream.is_open())  {                     // tests for openness
    cout << "Invalid file name. Please re-enter the name of the file to use: ";
    cin >> name;                                    // loop for correct file
    instream.open(name.c_str());
  }  
  
  //__________________________________
  //  bulletproofing
  int counter = 0;
  while(info != "GAMBIT" && counter < 10) {
    instream >> info;
    counter ++;
  }
  if (counter == 10 ) {
    cout << " The file is not recognized as a gambit file " << endl;
    cout << " Now exiting " << endl;
    exit (1);
  }
  
  //__________________________________
  //  find number of points and number of elements
  while(info != "NDFVL") {
    instream >> info;
  }
  instream >> numNodes;
  instream >> numElements;
  cout << "I've found "<< numNodes << " nodes and "
       << numElements<< " elements"<<endl;   
  
  //__________________________________
  // Find the nodal coordinates
  while(info != "COORDINATES"){
    instream >> info;
  }
  instream >> info;
  
  double nodeCoords[numNodes+1][4];
  for(int n=1; n<=numNodes; n++){
    for(int i = 0; i<=4; i++ ){
      nodeCoords[n][i] = -999;      // initialize to some obscure value
    }
  }
  // reads in node coordinates
  for(int x=1;x<=numNodes;x++){
    instream >> nodeCoords[x][0];  // this is really the node index
    instream >> nodeCoords[x][1];
    instream >> nodeCoords[x][2];
    instream >> nodeCoords[x][3];
    
    //__________________________________
    // bulletproofing
    if (nodeCoords[x][0] != x) {
      cout << " E R R O R : I've misread the Node Coordinates data "
	   << nodeCoords[x][0] << "   "<< nodeCoords[x][1] << "  "
	   << nodeCoords[x][2] << "   "<< nodeCoords[x][3] << endl;
      exit(1);
    } 
    if (!finite(nodeCoords[x][0]) || !finite(nodeCoords[x][1]) ||
	!finite(nodeCoords[x][2]) || !finite(nodeCoords[x][3]) ) {
      cout << " E R R O R :I've detected a number that isn't finite"<<endl;
      cout << "Node " << nodeCoords[x][0] << " " << nodeCoords[x][1] << " " <<
	nodeCoords[x][2] << " " << nodeCoords[x][3] << " " <<endl; 
      cout << " Now exiting "<<endl;
      exit(1);        
    } 
  } 
  
  //__________________________________
  //  read in element data
  counter = 0;
  while(info !="ELEMENTS/CELLS" && counter < 10){
    instream >> info;
    counter ++;
  }
  if(counter == 10 ) {
    cout << " The file is not recognized as a gambit file " << endl;
    cout << " Now exiting " << endl;
    exit (1);
  }   
  instream >> info;
  
  int nodeIndx[numElements+1][4];
  
  for(int y=1;y<=numElements;y++){
    instream >> nodeIndx[y][0];  // this is really element number
    instream >> info;
    instream >> info1;
    instream >> nodeIndx[y][1];
    instream >> nodeIndx[y][2];
    instream >> nodeIndx[y][3];
    
    //__________________________________
    // bulletproofing
    if (info != "3" || info1 != "3") {
      cout << " E R R O R : The input mesh is not triangulated surface "<< endl;
      cout << " Now exiting " << endl;
      exit(1);
    }
    if (nodeIndx[y][0] != y) {
      cout << " E R R O R : I've misread the element data "
	   << nodeIndx[y][0] << "  "<< info <<"  "<<info1
	   << "   "<< nodeIndx[y][1] << "  "<< nodeIndx[y][2]
	   << "   "<< nodeIndx[y][3] << endl;
      exit(1);
    }
    
    if (!finite(nodeIndx[y][0]) || !finite(nodeIndx[y][1]) || 
	!finite(nodeIndx[y][2]) || !finite(nodeIndx[y][3]) || 
	!finite(nodeIndx[y][4]) ) {
      cout << "I've detected a number that isn't finite"<<endl;
      cout << nodeIndx[y][0] << " " << nodeIndx[y][1] << " " <<
	nodeIndx[y][2] << " " << nodeIndx[y][3] <<endl; 
      cout << " Now exiting "<<endl;
      exit(1);        
    } 
  }
  
  //__________________________________
  //  spew out what I've found
#if 1
  cout << "Node Coordinates"<<endl;
  for(int x=1;x<=numNodes;x++){
    pts_stream << nodeCoords[x][0] << " " << nodeCoords[x][1] << " " 
	       << nodeCoords[x][2] << " " << nodeCoords[x][3] << endl; 
  }  
  
  cout << "connectivity "<<endl;
  for(int y=1;y<=numElements;y++){
    tri_stream << nodeIndx[y][0] << " " << nodeIndx[y][1] << " "
	       << nodeIndx[y][2] << " " << nodeIndx[y][3] << " " << endl;
  }  
#endif 
  
  cout << "I've successfully read in the file  " << endl;
  instream.close();
  
  return 0;
}
