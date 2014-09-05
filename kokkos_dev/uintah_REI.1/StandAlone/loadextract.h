#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <map>

#include <Packages/Uintah/StandAlone/KInpReader.h>

using namespace std;
using namespace SCIRun;

typedef struct 
{
	std::vector<unsigned int> elem;
  //unsigned int nodes[4];
	unsigned int node_id[4];
  double x[2][3];
} FACE;

typedef struct 
{
	double time;
	double pres;
} TIME_PRES;

class loadextract //this is a preprocessor for dyna
{
  public:

   loadextract(std::string kfilename, double pconv,
               double lconv1, double lconv2);
   ~loadextract(){};

   void extract(int argc, char** argv);
   void output();

   double presconv;
   double lengthconv1;
   double lengthconv2;

  protected:

  private:

   k_inp_reader kreader;
   template<class T>
      void printData(
               DataArchive* archive, string& variable_name, const Uintah::TypeDescription* variable_type,
               int material, const bool use_cellIndex_file, int levelIndex,
               IntVector& var_start, IntVector& var_end, vector<IntVector> cells,
               unsigned long time_start, unsigned long time_end, ostream& out);

        map<unsigned int, Node*> nodes_m;

        map<string, FACE> faces_m;

        vector<FACE> faces_shell;

	vector<FACE*> faces;

	vector<vector<TIME_PRES> > load_curve;
        int iface_shell;
};



	
