#include "/home/collab/sshankar/SVN/SCIRun/src/Packages/Uintah/StandAlone/tools/uda2nrrd/particleData.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include <iostream>

using namespace std;

int main()
{
        char *error;
	vector<string> arr2d;

	timeStep* (*processData)(int, char[][128], int, bool);	
	udaVars* (*getVarList)(const string&);
	int* (*getTimeSteps)(const string&);
	double* (*getBBox)(const string&, int);

	int a = 1;

	void *libHandle = dlopen("/home/collab/sshankar/SVN/SCIRun/build/lib/libPackages_Uintah_StandAlone_tools_uda2nrrd.so", RTLD_NOW);
	if (!libHandle)
	{
		printf("%s\n", dlerror());
		exit(1);
	}

	*(void **)(&processData) = dlsym(libHandle, "processData");
	if((error = dlerror()) != NULL)
	{
		printf("%s\n", error);
		exit(1);
	}

	*(void **)(&getVarList) = dlsym(libHandle, "getVarList");
	if((error = dlerror()) != NULL)
	{
		printf("%s\n", error);
		exit(1);
	}

	*(void **)(&getBBox) = dlsym(libHandle, "getBBox");
	if((error = dlerror()) != NULL)
	{
		printf("%s\n", error);
		exit(1);
	}

	arr2d.push_back( "uda2nrrd"); // anything will do
	arr2d.push_back( "-uda" );
	arr2d.push_back( "/home/collab/sshankar/csafe_data/big.uda");
	arr2d.push_back( "-v");
	arr2d.push_back( "tempSolid_CC");
	arr2d.push_back( "-o");
	arr2d.push_back( "test"); // anything will do

	timeStep *timeStepObjPtr = (*processData)(7, arr2d, 2, true);
	timeStep &timeStepObj = *timeStepObjPtr;

	// variables& varCollnRef = *(timeStepObj.varColln);
	
	// cout << dB.size() << endl;

	// cellVals& cellValColln = *(timeStepObj.cellValColln);
	// variables& varCollnRef = *(timeStepObj.varColln);
	// cout << cellValColln.x << " " << cellValColln.y << " " << cellValColln.z << endl;
	
	/*if (cellValColln.cellValVec != NULL)
	  cout << "Correct\n";*/

	/*for (int i = 0; i < cellValColln.size(); i++) {
	  cout << cellValColln[i] << endl;
	}*/
	
	/*for (int i = 0; i < varCollnRef.size(); i++) {
		variable& varRef = varCollnRef[i];
		unknownData& dataRef = varRef.data;
		for (int j = 0; j < dataRef.size(); j++)
			cout << dataRef[j].name << " " << dataRef[j].value << " ";
		cout << endl;
	}*/

	/*string fileName("/home/collab/sshankar/csafe_data/erosion.uda");

	udaVars* udaVarsPtr = (*getVarList)(fileName);
	udaVars& udaVarsObj = *(udaVarsPtr);

	// int* timeSteps = (*getTimeSteps)(fileName);

	for (int i = 0; i < udaVarsPtr->size(); i++) {
	  cout << udaVarsObj[i] << endl;
	}

	// cout << *timeSteps << endl;

	int timeStepNo = 0;
	double* minMaxArr = (*getBBox)(fileName, timeStepNo);

	cout << minMaxArr[0] << " " << minMaxArr[5] << endl;*/

	dlclose(libHandle);
	return 0;
}
