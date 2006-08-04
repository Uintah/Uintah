
#include <Packages/Uintah/CCA/Components/Models/test/TableFactory.h>
#include <Packages/Uintah/CCA/Components/Models/test/ArchesTable.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;

TableInterface* TableFactory::readTable(const ProblemSpecP& params,
					const string& name)
{
  for (ProblemSpecP child = params->findBlock("table"); child != 0;
       child = child->findNextBlock("table")) {
    string tname;
    if(child->getAttribute("name", tname) && tname == name){
      string type;
      if(!child->getAttribute("type", type))
	throw ProblemSetupException("Cannot read table type from table", __FILE__, __LINE__);
      if(type == "Arches")
	return new ArchesTable(child);
      else
	throw ProblemSetupException("Unknown table type: "+type, __FILE__, __LINE__);
    }
  }
  throw ProblemSetupException("Cannot find table: "+name, __FILE__, __LINE__);
}

