
#include <Packages/Uintah/CCA/Components/Models/test/ArchesTable.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Time.h>
#include <iostream>
#include <fstream>

// TODO:  Interpolation could be a lot faster
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
    defaults.push_back(df);
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
  for(int i=0;i<deps.size();i++){
    Dep* dep = deps[i];
    if(dep->name == name)
      return i;
  }
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
  double start = Time::currentSeconds();
  // Read the index...
  ifstream in(filename.c_str());
  startline = true;
  if(!in)
    throw ProblemSetupException("file not found: "+filename);
  

  cerr << "Reading table\n";
  int nvars = getInt(in);
  cerr << "Reading " << nvars << " variables : ";

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
    if(num <= 2)
      throw InternalError("Table must have at least size 2 in each dimension");
    ind->weights.resize(num);
  }
  for(int i=0;i<nvars;i++)
    cerr << in_inds[i]->weights.size() << " ";
  cerr << '\n';
  long size = 1;
  for(int i=0;i<nvars;i++)
    size *= in_inds[i]->weights.size();

  int ndeps = getInt(in);
  vector<Dep*> in_deps(ndeps);
  for(int i=0;i<ndeps;i++){
    Dep* dep = new Dep;
    
    dep->name = getString(in);
    dep->data = new double[size];
    in_deps[i] = dep;
  }

  long stride = 1;
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
  int size2 = 1;
  for(int i=0;i<nvars-1;i++)
    size2 *= in_inds[i]->weights.size();
  int size3 = in_inds[nvars-1]->weights.size();
  for(int i=0;i<size2;i++){
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
      int idx = n[i];
      if(first){
	in_inds[i]->weights[idx] = tmp;
      } else {
	if(in_inds[i]->weights[idx] != tmp){
	  throw InternalError("Inconsistent table (1)\n");
	}
      }
    }
    for(int i=0;i<size3;i++){
      // Read last var ndep times
      double last = getDouble(in);
      for(int i=0;i<ndeps-1;i++){
	double tmp = getDouble(in);
	if(tmp != last){
	  throw InternalError("Inconsistent table(2)\n");
	}
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
	if(in_inds[nvars-1]->weights[n[nvars-1]] != last)
	  throw InternalError("Iconsistent table(3)\n");
      }

      // Read deps
      int index=0;
      for(int i=0;i<nvars;i++)
	index += in_inds[i]->offset[n[i]];
      for(int i=0;i<ndeps;i++){
	double value = getDouble(in);
	in_deps[i]->data[index] = value;
      }
      int s = nvars-1;
      while(s >= 0 && ++n[s] >= (int)in_inds[s]->weights.size()){
	n[s]=0;
	s--;
      }
    }
  }

  // Verify uniformness...
  for(int i=0;i<nvars;i++){
    Ind* ind = in_inds[i];
    if(ind->uniform){
      int n = ind->weights.size();
      double dx = (ind->weights[n-1]-ind->weights[0])/(n-1);
      if(dx <= 0)
	throw InternalError("Backwards table not supported");
      ind->dx = dx;
      for(int i=1;i<n;i++){
	double dd = ind->weights[i]-ind->weights[i-1];
	if(Abs(dd-dx)/dx > 1.e-5){
	  cerr << "dx0=" << dx << ", dx[" << i << "]=" << dd << '\n';
	  cerr << "weights[" << i << "]=" << ind->weights[i] << ", weights[" << i-1 << "]=" << ind->weights[i] << '\n';
	  cerr << "difference: " << dd-dx << '\n';
	  cerr << "relative difference: " << Abs(dd-dx)/dx << '\n';
	  throw InternalError("Table labeled as uniform, but non-uniform spacing found");
	}
      }
    }
  }

  // Verify that independent/dependent variables are available
  vector<int> input_indices(inds.size());
  for(int i=0;i<inds.size();i++)
    input_indices[i] = -1;
  for(int i=0;i<inds.size();i++){
    Ind* ind = inds[i];
    bool found = false;
    for(int j=0;j<in_inds.size();j++){
      if(in_inds[j]->name == ind->name){
	found=true;
	input_indices[i] = j;
	break;
      }
    }
    if(!found)
      throw InternalError(string("Independent variable: ")+ind->name+" not found");
  }

  // Convert the input indeps
  long newstride=1;
  for(int i=0;i<inds.size();i++){
    Ind* ind = inds[i];
    Ind* input_ind = in_inds[input_indices[i]];
    ind->name = input_ind->name;
    ind->weights = input_ind->weights;
    ind->uniform = input_ind->uniform;
    ind->dx = input_ind->dx;
    cerr << i << ", input " << input_indices[i] << ", name=" << ind-> name << ", dx=" << ind->dx << '\n';
    ind->offset.resize(ind->weights.size());
    for(int i=0;i<ind->weights.size();i++)
      ind->offset[i] = i*newstride;
    newstride *= ind->weights.size();
  }
  long newsize = newstride;
  // Down-slice the table if necessary

  int dim_diff = in_inds.size()-inds.size();
  long interp_size = 1<<dim_diff;
  long* idx = new long[interp_size];
  double* w = new double[interp_size];
  for(int j=0;j<interp_size;j++){
    idx[j] = 0;
    w[j] = 1;
  }

  // Find the axes to be eliminated
  long s = 1;
  for(int i=0;i<in_inds.size();i++){
    bool found=false;
    for(int j=0;j<inds.size();j++){
      if(in_inds[i]->name == inds[j]->name){
	found=true;
	break;
      }
    }
    if(!found){
      // Find the default value...
      bool found = false;
      double value;
      for(int j=0;j<defaults.size();j++){
	if(in_inds[i]->name == defaults[j]->name){
	  found=true;
	  value=defaults[j]->value;
	}
      }
      if(!found)
	throw InternalError("Default value for "+in_inds[i]->name+" not found");
      Ind* ind = in_inds[i];
      int l=0;
      int h=ind->weights.size()-1;
      if(value < ind->weights[l] || value > ind->weights[h])
	throw InternalError("Interpolate outside range of table");
      while(h > l+1){
	int m = (h+l)/2;
	if(value < ind->weights[m])
	  h=m;
	else
	  l=m;
      }
      long i0 = ind->offset[l];
      long i1 = ind->offset[h];
      double w0 = (value-ind->weights[l])/(ind->weights[h]-ind->weights[l]);
      double w1 = 1-w0;
      for(int j=0;j<interp_size;j++){
	if(j&s){
	  idx[j] += i1;
	  w[j] *= w0;
	} else {
	  idx[j] += i0;
	  w[j] *= w1;
	}
      }
    }
  }
  for(int i=0;i<deps.size();i++){
    Dep* dep = deps[i];
    Dep* inputdep = 0;
    for(int j=0;j<in_deps.size();j++){
      if(in_deps[j]->name == dep->name){
	inputdep = in_deps[j];
	break;
      }
    }
    if(!inputdep)
      throw InternalError(string("Dependent variable: ")+dep->name+" not found");
    cerr << "Downslicing: " << dep->name << '\n';
    dep->data = new double[newsize];

    for(int i=0;i<inds.size();i++)
      n[i]=0;

    for(int i=0;i<newsize;i++){
      double sum = 0;
      long iidx = 0;
      for(int j=0;j<inds.size();j++){
	Ind* ind = in_inds[input_indices[j]];
	iidx += ind->offset[n[j]];
      }

      for(int j=0;j<interp_size;j++){
	long index = iidx+idx[j];
        sum += w[j]*inputdep->data[index];
      }

      long oidx = 0;
      for(int j=0;j<inds.size();j++){
	oidx += inds[j]->offset[n[j]];
      }
      dep->data[oidx] = sum;
      int s = inds.size()-1;
      while(s >= 0 && ++n[s] >= (int)inds[s]->weights.size()){
	n[s]=0;
	s--;
      }
      
    }
  }
  file_read = true;
  double dt = Time::currentSeconds()-start;
  cerr << "Read and interpolated table in " << dt << " seconds\n";
}
    
void ArchesTable::interpolate(int index, CCVariable<double>& result,
			      const CellIterator& in_iter,
			      vector<constCCVariable<double> >& independents)
{
  Dep* dep = deps[index];
  int ni = inds.size();
  double* w = new double[ni];
  long* idx0 = new long[ni];
  long* idx1 = new long[ni];
  long ninterp = 1<<ni;

  for(CellIterator iter = in_iter; !iter.done(); iter++){
    for(int i=0;i<ni;i++){
      Ind* ind = inds[i];
      double value = independents[i][*iter];
      if(ind->uniform){
	double index = (value-ind->weights[0])/ind->dx;
	if(index < 0 || index >= ind->weights.size()){
	  cerr << "value=" << value << ", start=" << ind->weights[0] << ", dx=" << ind->dx << '\n';
	  cerr << "index=" << index << '\n';
	  throw InternalError("Interpolate outside range of table");
	}
	int idx = (int)idx;
	w[i] = index-idx;
	idx0[i] = ind->offset[idx];
	idx1[i] = ind->offset[idx+1];
      } else {
	int l=0;
	int h=ind->weights.size()-1;
	if(value < ind->weights[l] || value > ind->weights[h])
	  throw InternalError("Interpolate outside range of table");
	while(h > l+1){
	  int m = (h+l)/2;
	  if(value < ind->weights[m])
	    h=m;
	  else
	    l=m;
	}
	idx0[i] = ind->offset[l];
	idx1[i] = ind->offset[h];
	w[i] = (value-ind->weights[l])/(ind->weights[h]-ind->weights[l]);
      }
    }
    // Do the interpolation
    double sum = 0;
    for(long i=0;i<ninterp;i++){
      double weight = 1;
      long index = 0;
      for(int j=0;j<ni;j++){
	long mask = 1<<j;
	if(i & mask){
	  index += idx1[j];
	  weight *= (1-w[j]);
	} else {
	  index += idx0[j];
	  weight *= w[j];
	}
      }
      double value = dep->data[index] * weight;
      sum += value;
    }
    result[*iter] = sum;
  }
  delete[] idx0;
  delete[] idx1;
  delete[] w;  
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

  if(!isdigit(c) && c != '-' && c != '.')
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
  result.erase(result.find_last_not_of(" \t\n")+1);
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
    return false;
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

