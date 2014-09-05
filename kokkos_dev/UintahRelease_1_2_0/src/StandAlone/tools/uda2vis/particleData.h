/*

   The MIT License

   Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#pragma once

#include <string>
#include <vector>
using namespace std;


// Vector variables
class vecVal {
  public:
    string name;
    float x;
    float y;
    float z;
    vecVal(){};
    void operator=(const vecVal& obj) {
      // name.assign(obj.name);
      x = obj.x;
      y = obj.y;
      z = obj.z;
    }
    ~vecVal(){};
};

// Tensor variables
class tenVal {
  public:
    string name;
    double mat[3][3];
    tenVal(){};
    void operator=(const tenVal& obj) {
      // name.assign(obj.name);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
	  mat[i][j] = obj.mat[i][j];
	}
      }
    }
    ~tenVal(){};
};



// Level-Patch pair
class levelPatch {
  public:
    int levelNo;
    int noPatches;
    int rr[3];
    levelPatch(int level, int patches, int x, int y, int z) {
      levelNo = level;
      noPatches = patches;
      rr[0] = x;
      rr[1] = y;
      rr[2] = z;
    };
    ~levelPatch(){};
};

typedef vector<levelPatch> levelPatchVec;


// Patch collection
// 
class patchInfo {
  public: 
    int indexArr[6];
    int hiLoArr[6];
    double minMaxArr[6];
    int numCells;
    patchInfo(int* indexArrPtr, double* minMaxArrPtr, int *hiLoPtr, int nCells) {
      indexArr[0] = indexArrPtr[0];	
      indexArr[1] = indexArrPtr[1];	
      indexArr[2] = indexArrPtr[2];	
      indexArr[3] = indexArrPtr[3];	
      indexArr[4] = indexArrPtr[4];	
      indexArr[5] = indexArrPtr[5];

      minMaxArr[0] = minMaxArrPtr[0];
      minMaxArr[1] = minMaxArrPtr[1];
      minMaxArr[2] = minMaxArrPtr[2];
      minMaxArr[3] = minMaxArrPtr[3];
      minMaxArr[4] = minMaxArrPtr[4];
      minMaxArr[5] = minMaxArrPtr[5];

      hiLoArr[0] = hiLoPtr[0];
      hiLoArr[1] = hiLoPtr[1];
      hiLoArr[2] = hiLoPtr[2];
      hiLoArr[3] = hiLoPtr[3];
      hiLoArr[4] = hiLoPtr[4];
      hiLoArr[5] = hiLoPtr[5]; 						

      numCells = nCells;
    }
    ~patchInfo(){};
};

typedef vector<patchInfo> patchInfoVec;



class ParticleVariableRaw {
  public:
  int components;
  vector<float> values;
};



typedef vector<double> typeDouble;

class cellVals {
  public:
    int x, y, z, dim;
    typeDouble* cellValVec;
    cellVals() {
      cellValVec = NULL;
    };
    ~cellVals() {
      if (cellValVec) delete cellValVec;
    };
};	

typedef vector<string> udaVars;
typedef vector<int> varMatls;

class timeStep {
  public:
  ParticleVariableRaw partVar;
    cellVals* cellValColln;
    int no;
    string name;
    timeStep() {
      cellValColln = NULL;
    };
    ~timeStep() {
      if (cellValColln) delete cellValColln;  
    };
};

typedef vector<timeStep> udaData;
