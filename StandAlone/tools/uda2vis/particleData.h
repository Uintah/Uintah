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


#pragma once

#include <string>
#include <vector>
using namespace std;
  
enum DataType { UNKNOWN=-1, SCALAR=0, VECTOR=1, TENSOR=2 };

class nameVal {
 public:
	string name;
	float value;
	nameVal(){};
	~nameVal(){};
};

typedef vector<nameVal> unknownData;

// Vector variables
class vecVal {
 public:
	string name;
	float x;
	float y;
	float z;
	vecVal(){};
	~vecVal(){};
};

// Tensor variables
class tenVal {
 public:
	string name;
	double mat[3][3];
	tenVal(){};
	~tenVal(){};
};

typedef vector<vecVal> vecValData;
typedef vector<tenVal> tenValData;

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
   patchInfo(int* indexArrPtr, double* minMaxArrPtr, int *hiLoPtr) {
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
     // memcpy(indexArr, indexArrPtr, sizeof(indexArrPtr));
	 // memcpy(minMaxArr, minMaxArrPtr, sizeof(minMaxArrPtr));
   }
   ~patchInfo(){};
};

typedef vector<patchInfo> patchInfoVec;

class variable {
 public:
	float x;
	float y;
	float z;
	unknownData data;
    vecValData vecData;
    tenValData tenData;
	variable() {
	  // vecData = NULL;
	};
	~variable() { 
	  // if (vecData) delete vecData;
	};
	variable(const variable &obj)
	{
		this->x = obj.x;
		this->y = obj.y;
		this->z = obj.z;
		this->data = obj.data;	
		this->vecData = obj.vecData;
		this->tenData = obj.tenData;
	}
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

typedef vector<variable> variables;
typedef vector<string> udaVars;
typedef vector<int> varMatls;

class timeStep {
 public:
	variables* varColln;
	cellVals* cellValColln;
	int no;
	string name;
	timeStep() {
	  varColln = NULL;
	  cellValColln = NULL;
	};
	~timeStep() {
	  if (varColln) delete varColln;
	  if (cellValColln) delete cellValColln;  
    };
};

typedef vector<timeStep> udaData;
