
#include <Packages/Uintah/CCA/Components/ICE/Combined/CombinedFactory.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <string>

using namespace Uintah;

PropertyBase* CombinedFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP child = ps->findBlock("Combined");
  if(!child)
    return 0; // The tag might not be there
  std::string type;
  if(!child->getAttribute("type",type))
    throw ProblemSetupException("No type for Combined", __FILE__, __LINE__); 

#if 0  
  if (type == "cantera") 
    return(scinew CanteraProperties(child));
  else
#endif
    throw ProblemSetupException("Unknown Combined Type ("+type+")", __FILE__, __LINE__);
}
