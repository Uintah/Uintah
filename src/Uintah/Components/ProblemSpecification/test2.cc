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
  prob_spec = reader.readInputFile("test.ups");

  cout << "works after reading file . . ." << endl;

  ProblemSpecP testps = prob_spec->findBlock("Time");

  cout << "Works after finding Time block" << endl;
  exit(1);
}

