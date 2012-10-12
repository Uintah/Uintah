/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// To compile:  g++ -o gambitFileReader gambitFileReader.cc
// This will take in a data file from GAMBIT
//////////////////////////////////////////////////

#include <iostream>                    
#include <string>                      
#include <fstream>  
#include <vector>                   
#include <cmath>
#include <cstdlib>
#define d_SMALL_NUM 1e-100;
using namespace std;

#ifdef _WIN32
#include <cfloat>
#define finite _finite
#endif

int x, y;

struct node_coord {
  double x,y,z;
};

struct elem_coord {
  int n1,n2,n3;
};

typedef struct node_coord node_coord;
typedef struct elem_coord elem_coord;

int main()
{
  ifstream instream;       
  ofstream pts_stream,tri_stream;
  
  int numNodes = 0;
  int numElements = 0;
  string name,out_name, info, info1, info2;
  cout << endl << "**********************************************" << endl;
  cout << "Input the name of the Gambit output file: ";
  cin >> name;

  cout << "Input the output file name for storing points and surfaces: ";
  cin >> out_name;
 
  instream.open(name.c_str());                       // opens files name
 
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
  
  instream>> info1;
  instream>> info2;
  
  if (counter == 10       || 
       info  != "GAMBIT"  ||
       info1 != "NEUTRAL" ||
       info2 != "FILE"  ) {
    cout << " The file is not recognized as a gambit neutral (neu) file " << endl;
    cout << " Now exiting " << endl;
    exit (1);
  }
  
  string pts_name = out_name + ".pts";
  string tri_name = out_name + ".tri";
  pts_stream.open(pts_name.c_str());
  tri_stream.open(tri_name.c_str());
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
  
  vector<struct node_coord> nodeCoords(numNodes);
  for(int n=0; n<numNodes; n++){
      nodeCoords[n].x = -999;      // initialize to some obscure value
      nodeCoords[n].y = -999;      // initialize to some obscure value
      nodeCoords[n].z = -999;      // initialize to some obscure value
  }
  int node_num;
  // reads in node coordinates
  for(int x=0;x<numNodes;x++){
    instream >> node_num;
    instream >> nodeCoords[x].x;
    instream >> nodeCoords[x].y;
    instream >> nodeCoords[x].z;
    
    //__________________________________
    // bulletproofing
#if 1
    if (node_num != x+1) {
      cout << " E R R O R : I've misread the Node Coordinates data "
	   << nodeCoords[x].x << "  "
	   << nodeCoords[x].y << "   "<< nodeCoords[x].z << endl;
      exit(1);
    } 
    if (!finite(nodeCoords[x].x) ||
	!finite(nodeCoords[x].y) || !finite(nodeCoords[x].z) ) {
      cout << " E R R O R :I've detected a number that isn't finite"<<endl;
      cout << " " << nodeCoords[x].x << " " <<
	nodeCoords[x].y << " " << nodeCoords[x].z << " " <<endl; 
      cout << " Now exiting "<<endl;
      exit(1);        
    } 
  } 
#endif
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
  
  vector<elem_coord> nodeIndx(numElements);
  
  for(int y=0;y<numElements;y++){
    instream >> node_num;  // this is really element number
    instream >> info;
    instream >> info1;
    instream >> nodeIndx[y].n1;
    instream >> nodeIndx[y].n2;
    instream >> nodeIndx[y].n3;
    
    //__________________________________
    // bulletproofing
    if (info != "3" || info1 != "3") {
      cout << " E R R O R : The input mesh is not triangulated surface "<< endl;
      cout << " Now exiting " << endl;
      exit(1);
    }
    if (node_num != y+1) {
      cout << " E R R O R : I've misread the element data "
	   << info <<"  "<<info1
	   << "   "<< nodeIndx[y].n1 << "  "<< nodeIndx[y].n2
	   << "   "<< nodeIndx[y].n3 << endl;
      exit(1);
    }
    
    if (!finite(nodeIndx[y].n1) || 
	!finite(nodeIndx[y].n2) || !finite(nodeIndx[y].n3)) {
      cout << "I've detected a number that isn't finite"<<endl;
      cout << nodeIndx[y].n1 << " " <<
	nodeIndx[y].n2 << " " << nodeIndx[y].n3 <<endl; 
      cout << " Now exiting "<<endl;
      exit(1);        
    } 
  }
  
  //__________________________________
  //  spew out what I've found
  cout << "Node Coordinates"<<endl;
  for(int x=0;x<numNodes;x++){
    pts_stream << nodeCoords[x].x << " " 
	       << nodeCoords[x].y << " " << nodeCoords[x].z << endl; 
  }  
  
  // Input file is 1 based, changing it to zero based.
  cout << "connectivity "<<endl;
  for(int y=0;y<numElements;y++){
    tri_stream << nodeIndx[y].n1-1 << " "
	       << nodeIndx[y].n2-1 << " " << nodeIndx[y].n3-1 << " " << endl;
  }  
  
  cout << "I've successfully read in the file  " << endl;
  instream.close();
  
  return 0;
}
