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
