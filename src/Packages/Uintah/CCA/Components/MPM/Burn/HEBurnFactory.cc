#include "HEBurnFactory.h"
#include "NullHEBurn.h"
#include "SimpleHEBurn.h"
#include <Core/Malloc/Allocator.h>
#include <string>
using std::cerr;


using namespace Uintah;

HEBurn* HEBurnFactory::create(ProblemSpecP& ps)
{


   for (ProblemSpecP child = ps->findBlock("burn"); child != 0;
	child = child->findNextBlock("burn")) {
      std::string burn_type;
      child->require("type",burn_type);
      cerr << "burn_type = " << burn_type << std::endl;
    
      if (burn_type == "null")
	 return(scinew NullHEBurn(child));
      
      else if (burn_type == "simple")
	 return(scinew SimpleHEBurn(child));

      else {
	 cerr << "Unknown Burn Type R (" << burn_type << ")" << std::endl;;
	 //      exit(1);
      }
    
   }
   return 0;
}

