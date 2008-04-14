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

class variable {
 public:
	float x;
	float y;
	float z;
	unknownData data;
	// float volume; // Could replace these by a vector having a <string name, float value> pair as data items 
	// float stress; // 
	variable(){};
	~variable(){};
	variable(const variable &obj)
	{
		this->x = obj.x;
		this->y = obj.y;
		this->z = obj.z;
		this->data = obj.data;	
		// this->volume = obj.volume;
		// this->stress = obj.stress;	
	}
};

typedef vector<double> typeDouble;

class cellVals {
 public:
	int x, y, z;
	typeDouble* cellValVec;
	cellVals() {
	  cellValVec = NULL;
	};
	~cellVals() {
	  if (cellValVec) delete cellValVec;
	};
};	

typedef vector<variable> variables;
// typedef vector<double> cellVals;
typedef vector<string> udaVars;

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
