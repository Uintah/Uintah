#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <string>
#include <iostream>

using std::cout;
using std::endl;
using std::cerr;
using namespace Uintah;

int main()
{

  ProblemSpecReader reader("input.ups");
  
  ProblemSpecP prob_spec;
  prob_spec = reader.readInputFile();

  cout << "works after reading file . . ." << endl;

  ProblemSpecP uintah_prob_spec = prob_spec->findBlock("Uintah_specification");
    
  ProblemSpecP meta_prob_spec = prob_spec->findBlock("Meta");

  ProblemSpecP time_prob_spec = prob_spec->findBlock("Time");

  ProblemSpecP cfd_prob_spec = prob_spec->findBlock("CFD");

  ProblemSpecP ice_prob_spec = cfd_prob_spec->findBlock("ICE");

  ProblemSpecP mat_prob_spec = prob_spec->findBlock("MaterialProperties");
  
  ProblemSpecP mpm_mat_ps = mat_prob_spec->findBlock("MPM");

  cout << "Works after finding ICE block" << endl;

  std::string title;
  meta_prob_spec->require("title",title);
  cout << "title is " << title << endl;

  double maxTime;
  time_prob_spec->require("maxTime",maxTime);
  cout << "maxTime is " << maxTime << endl;
  
  double initTime;
  time_prob_spec->require("initTime",initTime);
  cout << "initTime is " << initTime << endl;

  double delt_min;
  time_prob_spec->require("delt_min",delt_min);
  cout << "delt_min is " << delt_min << endl;

  double delt_max;
  time_prob_spec->require("delt_max",delt_max);
  cout << "delt_max is " << delt_max << endl;

  ProblemSpecP ice_mat_prob_spec = 
    ice_prob_spec->findBlock("material_properties");

  double viscosity;
  ice_mat_prob_spec->require("viscosity",viscosity);
  cout << "viscosity is " << viscosity << endl;

  // Here is an example of picking out multiple tags from within a block

  for (ProblemSpecP mat_ps = mpm_mat_ps->findBlock("material"); mat_ps != 0;
       mat_ps = mat_ps->findNextBlock("material") ) {
    std::string material_type;
    mat_ps->require("material_type", material_type);
    cout << "material_type is " <<  material_type << endl;
   
  }
  for (ProblemSpecP mat_ps = mpm_mat_ps->findBlock(); mat_ps != 0;
       mat_ps = mat_ps->findNextBlock() ) {
    cout << "name is " << mat_ps->getNodeName() << endl;
  }

  prob_spec->d_node->getOwnerDocument()->release();
  exit(1);
}

