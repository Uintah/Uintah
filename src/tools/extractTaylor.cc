/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

// Find the current location of a set of points whose initial
// position is known (from puda) (for the Taylor impact test)
using namespace std;

bool inbox(double px, double py, double pz, double xmin, double xmax,
	   double ymin, double ymax, double zmin, double zmax)
{
  if (px > xmin && py > ymin && pz > zmin && px < xmax && py < ymax
      && pz < zmax) return true;
  return false;
}

int main(int argc, char** argv)
{
   // Check if the number of arguments is correct
   if (argc != 3) {
     cerr << "Usage: extractTaylor <int_pos_file> <final_pos_file>" 
          << endl;
     exit(0);
   }

   // Parse arguments 
   string posFileName = argv[1];
   string finposFileName = argv[2];
      
   // Open the files
   ifstream posFile(posFileName.c_str());
   if(!posFile) {
     cerr << "File " << posFileName << " can't be opened." << endl;
     exit(0);
   }
   string pidFileName = posFileName+".pid";
   ofstream pidFile(pidFileName.c_str());

   // Read in the initial particle location
   double rad, height;
   cout << "\n Enter radius and height ";
   cin >> rad >> height;
   cout << rad << " " << height << endl;
   double dx;
   cout << "\n Enter width of search box ";
   cin >> dx;
   cout << dx << endl;

   // Create an array for storing the PIDs
   vector<int64_t> pidVec;

   // Read the header 
   posFile.ignore(1000,'\n');

   int64_t pID;
   int patch, mat;
   double time, x, y, z;
   double xmin, ymin, xmax, ymax, zmin, zmax;
   while (posFile >> time >> patch >> mat >> pID >> x >> y >> z){

     // Create the width box (top)
     xmin = -dx; ymin = height - dx; zmin = -dx;
     xmax = rad + dx; ymax = height + dx; zmax = dx;
     if (inbox(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax)){
       //cout << " pID = " << pID << " [" << x <<","<<y<<","<<z<<"]\n";
       pidFile << pID << " " << x << " " << y << " " << z << endl;
       pidVec.push_back(pID);
     }

     // Create the height box
     xmin = rad - dx; ymin = -dx; zmin = -dx;
     xmax = rad + dx; ymax = height + dx; zmax = dx;
     if (inbox(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax)){
       //cout << " pID = " << pID << " [" << x <<","<<y<<","<<z<<"]\n";
       pidFile << pID << " " << x << " " << y << " " << z << endl;
       pidVec.push_back(pID);
     }

     // Create the width box (bottom)
     xmin = -dx; ymin = - dx; zmin = -dx;
     xmax = rad + dx; ymax = dx; zmax = dx;
     if (inbox(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax)){
       //cout << " pID = " << pID << " [" << x <<","<<y<<","<<z<<"]\n";
       pidFile << pID << " " << x << " " << y << " " << z << endl;
       pidVec.push_back(pID);
     }

   }
   posFile.close();

   // Open the files
   ifstream finposFile(finposFileName.c_str());
   if(!finposFile) {
     cerr << "File " << finposFileName << " can't be opened." << endl;
     exit(0);
   }
   string finpidFileName = finposFileName+".pid";
   ofstream finpidFile(finpidFileName.c_str());
   // Read the header 
   finposFile.ignore(1000,'\n');

   int numPID = pidVec.size();
   while (finposFile >> time >> patch >> mat >> pID >> x >> y >> z){
     for (int ii = 0; ii < numPID; ++ii) {
       if (pID == pidVec[ii])
         //cout << " pID = " << pID << " [" << x <<","<<y<<","<<z<<"]\n";
         finpidFile << pID << " " << x << " " << y << " " << z << endl;
     } 
   }
   finposFile.close();

}
