#include <Uintah/Components/ProblemSpecification/ProblemSpecReader.h>
#include <string>
#include <iostream>

using std::cout;
using std::endl;

using Uintah::Interface::ProblemSpec;
using Uintah::Interface::ProblemSpecP;


int main()
{

  ProblemSpecReader reader;
  
  ProblemSpecP prob_spec;
  prob_spec = reader.readInputFile("input1.ups");

  cout << "works after reading file . . ." << endl;
  
  ProblemSpecP meta_prob_spec = prob_spec->findBlock("Meta");

  ProblemSpecP time_prob_spec = prob_spec->findBlock("Time");

  ProblemSpecP cfd_prob_spec = prob_spec->findBlock("CFD");

  ProblemSpecP ice_prob_spec = cfd_prob_spec->findBlock("ICE");

   cout << "Works after finding ICE block" << endl;

  string title;
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

  ProblemSpecP mat_prob_spec = ice_prob_spec->findBlock("material_properties");

  double viscosity;
  mat_prob_spec->require("viscosity",viscosity);
  cout << "viscosity is " << viscosity << endl;

  
 
  

  exit(1);
}

