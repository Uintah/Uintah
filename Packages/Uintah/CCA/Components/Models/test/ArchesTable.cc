
#include <Packages/Uintah/CCA/Components/Models/test/ArchesTable.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <iostream>
#include <fstream>
using namespace std;

using namespace Uintah;

ArchesTable::ArchesTable(ProblemSpecP& params)
{
  params->require("filename", filename);
  for (ProblemSpecP child = params->findBlock("defaultValue"); child != 0;
       child = child->findNextBlock("defaultValue")) {
    DefaultValue* df = new DefaultValue;
    if(!child->getAttribute("name", df->name))
      throw ProblemSetupException("No name for defaultValue");
    child->get(df->value);
  }
  file_read = false;
}

ArchesTable::~ArchesTable()
{
  for(int i=0;i<(int)inds.size();i++)
    delete inds[i];
  for(int i=0;i<(int)deps.size();i++){
    delete[] deps[i]->data;
    delete deps[i];
  }
}

int ArchesTable::addDependentVariable(const string& name)
{
  ASSERT(!file_read);
  Dep* dep = new Dep;
  dep->name = name;
  dep->data = 0;
  deps.push_back(dep);
  return (int)deps.size()-1;
}

void ArchesTable::addIndependentVariable(const string& name)
{
  ASSERT(!file_read);
  Ind* ind = new Ind;
  ind->name = name;
  inds.push_back(ind);
}


void ArchesTable::setup()
{
  // Read the index...
  ifstream in(filename.c_str());
  if(!in)
    throw ProblemSetupException("file not found: "+filename);
  

  int nvars;
  in >> nvars;
  for(int i=0;i<nvars;i++){
    string name;
    in >> name; // comma delim
  }
  // Down-slice the table if necessary
  cerr << "setup not done\n";
}
    
void ArchesTable::interpolate(int index, CCVariable<double>& result,
			      vector<constCCVariable<double> >& independents)
{
  cerr << "interpolate not done\n";
}

