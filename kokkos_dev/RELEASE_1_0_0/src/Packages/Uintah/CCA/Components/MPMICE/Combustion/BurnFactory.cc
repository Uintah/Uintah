#include "BurnFactory.h"
#include "NullBurn.h"
#include "SimpleBurn.h"
#include <Core/Malloc/Allocator.h>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;

using namespace Uintah;

Burn* BurnFactory::create(ProblemSpecP& ps)
{


   for (ProblemSpecP child = ps->findBlock("burn"); child != 0;
	child = child->findNextBlock("burn1")) {
      std::string burn_type;
      child->require("type",burn_type);
      cerr << "burn_type = " << burn_type << std::endl;
    
      if (burn_type == "null")
	 return(scinew NullBurn(child));
      
      else if (burn_type == "simple")
	 return(scinew SimpleBurn(child));

      else {
	 cerr << "Unknown Burn Type R (" << burn_type << ")" << std::endl;;
	 //      exit(1);
      }
    
   }
   return 0;
}

