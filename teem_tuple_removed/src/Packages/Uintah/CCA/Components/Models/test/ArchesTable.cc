
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
  startline = true;
  if(!in)
    throw ProblemSetupException("file not found: "+filename);
  

  cerr << "Reading table\n";
  int nvars = getInt(in);
  cerr << "Reading " << nvars << " variables\n";

  vector<Ind*> in_inds(nvars);
  for(int i=0;i<nvars;i++){
    Ind* ind = new Ind;
    ind->name = getString(in);
    in_inds[i] = ind;
  }
  for(int i=0;i<nvars;i++){
    Ind* ind = in_inds[i];
    ind->uniform = getBool(in);
  }
  for(int i=0;i<nvars;i++){
    Ind* ind = in_inds[i];
    int num = getInt(in);
    ind->weights.resize(num);
  }
  cerr << "Variables: ";
  for(int i=0;i<nvars;i++)
    cerr << in_inds[i]->weights.size() << " ";
  cerr << '\n';
  int ndeps;
  int size = 1;
  for(int i=0;i<nvars;i++)
    size *= in_inds[i]->weights.size();

  vector<Dep*> in_deps(ndeps);
  for(int i=0;i<ndeps;i++){
    Dep* dep = new Dep;
    dep->name = getString(in);
    dep->data = new double[size];
    in_deps[i] = dep;
  }

  int stride = 1;
  for(int s=nvars-1;s>=0;s--){
    Ind* ind = in_inds[s];
    int n = ind->weights.size();
    ind->offset.resize(n);
    for(int i=0;i<n;i++)
      ind->offset[i] = stride*i;
    stride *= n;
  }

  vector<int> n(nvars);
  for(int i=0;i<nvars;i++)
    n[i]=0;
  for(int i=0;i<size;i++){
    int s = nvars-1;
    while(n[s]++ >= (int)in_inds[s]->weights.size()){
      n[s]=0;
      s--;
    }
    for(int i=0;i<nvars;i++)
      cerr << n[i] << " ";
    cerr << '\n';

    // Read n-1 vars...
    for(int i=0;i<nvars-1;i++){
      double tmp = getDouble(in);
      bool first=true;
      for(int j=0;j<i;j++){
	if(n[j] != 0){
	  first=false;
	  break;
	}
      }
      if(first){
	in_inds[i]->weights[i] = tmp;
      } else {
	if(in_inds[i]->weights[i] != tmp){
	  throw InternalError("Inconsistent table (1)\n");
	}
      }
    }
    // Read last var ndep times
    double last = getDouble(in);
    for(int i=0;i<ndeps;i++){
      double tmp = getDouble(in);
      if(tmp != last)
	throw InternalError("Inconsistent table(2)\n");
    }
    bool first=true;
    for(int j=0;j<nvars-1;j++){
      if(n[j] != 0){
	first=false;
	break;
      }
    }
    if(first){
      in_inds[nvars-1]->weights[n[nvars-1]] = last;
    } else {
      if(in_inds[nvars-1]->weights[n[nvars-1]])
	throw InternalError("Iconsistent table(2)\n");
    }

    // Read deps
    int index=0;
    for(int i=0;i<nvars;i++)
      index += in_inds[i]->offset[n[i]];
    for(int i=0;i<ndeps;i++){
      double value = getDouble(in);
      in_deps[i]->data[index] = value;
    }
  }

  // Verify uniformness...

  // Down-slice the table if necessary
  cerr << "setup not done\n";
}
    
void ArchesTable::interpolate(int index, CCVariable<double>& result,
			      vector<constCCVariable<double> >& independents)
{
  cerr << "interpolate not done\n";
}

void ArchesTable::error(istream& in)
{
  string s;
  getline(in, s);
  throw InternalError("Error parsing table, text follows: "+s);
}

int ArchesTable::getInt(istream& in)
{
  if(startline)
    skipComments(in);
  
  eatWhite(in);

  int c = in.get();

  if(!isdigit(c))
    error(in);
  in.unget();
  int tmp;
  in >> tmp;
  if(!in)
    error(in);

  c = in.get();
  if(!isspace(c))
    error(in);
  return tmp;
}

double ArchesTable::getDouble(istream& in)
{
  eatWhite(in);
  if(startline)
    skipComments(in);  
  eatWhite(in);

  int c = in.get();

  if(!isdigit(c))
    error(in);
  in.unget();
  double tmp;
  in >> tmp;
  if(!in)
    error(in);

  c = in.get();
  if(!isspace(c))
    error(in);
  return tmp;
}

string ArchesTable::getString(istream& in)
{
  eatWhite(in);
  if(startline)
    skipComments(in);
  eatWhite(in);

  string result;
  eatWhite(in);
  for(;;){
    int c = in.get();
    if(c == ',' || c == '\n')
      break;
    result.push_back(c);
  }
  cerr << "Should trim space at end of strings!\n";
  return result;
}

bool ArchesTable::getBool(istream& in)
{
  eatWhite(in);
  if(startline)
    skipComments(in);
  eatWhite(in);
  int c = in.get();
  if(c == 'Y' || c == 'y')
    return true;
  else if(c == 'N' || c == 'n')
    return true;
  else
    throw InternalError("Error parsing yes/no bool value in table");
}

void ArchesTable::skipComments(istream& in)
{
  eatWhite(in);
  int c = in.get();
  while(c == '#'){
    while(c != '\n')
      c = in.get();
    eatWhite(in);
    c=in.get();
  }
  in.unget();
}

void ArchesTable::eatWhite(istream& in)
{
  int c = in.get();
  while(isspace(c))
    c = in.get();
  in.unget();
}

