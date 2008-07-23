#pragma once

#include <string>
#include <vector>
using namespace std;

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
   double minMaxArr[6];
   patchInfo(int* indexArrPtr, double* minMaxArrPtr) {
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
