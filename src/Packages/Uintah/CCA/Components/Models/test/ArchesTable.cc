
#include <Packages/Uintah/CCA/Components/Models/test/ArchesTable.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Time.h>
#include <iostream>
#include <fstream>

#define MAXINDEPENDENTS 100

// TODO:  Interpolation could be a lot faster
// TODO: parentheses in expressions, other ops in expressions
using namespace std;

using namespace Uintah;

ArchesTable::ArchesTable(ProblemSpecP& params)
{
  file_read = false;
  params->require("filename", filename);
  for (ProblemSpecP child = params->findBlock("defaultValue"); child != 0;
       child = child->findNextBlock("defaultValue")) {
    DefaultValue* df = new DefaultValue;
    if(!child->getAttribute("name", df->name))
      throw ProblemSetupException("No name for defaultValue");
    child->get(df->value);
    defaults.push_back(df);
  }
  for (ProblemSpecP child = params->findBlock("constantValue"); child != 0;
       child = child->findNextBlock("constantValue")) {
    Dep* dep = new Dep(Dep::ConstantValue);
    if(!child->getAttribute("name", dep->name))
      throw ProblemSetupException("No name for constantValue");
    child->get(dep->constantValue);
    deps.push_back(dep);
  }
  for (ProblemSpecP child = params->findBlock("derivedValue"); child != 0;
       child = child->findNextBlock("derivedValue")) {
    Dep* dep = new Dep(Dep::DerivedValue);
    if(!child->getAttribute("name", dep->name))
      throw ProblemSetupException("No expression for derivedValue");
    string expr;
    child->get(expr);
    string::iterator beg = expr.begin();
    string::iterator end = expr.end();
    dep->expression = parse_addsub(beg, end);
    if(beg != end || !dep->expression){
      cerr << "expression = " << dep->expression << '\n';
      if(beg != end)
        cerr << "next=" << *beg << '\n';
      cerr << "Error parsing expression:\n" << expr << '\n';
      for(string::iterator skip = expr.begin(); skip != beg; skip++)
        cerr << ' ';
      cerr << "^\n";
      throw ProblemSetupException("Error parsing expression");
    }
    deps.push_back(dep);
  }
}

ArchesTable::~ArchesTable()
{
  for(int i=0;i<(int)inds.size();i++)
    delete inds[i];
  for(int i=0;i<(int)deps.size();i++){
    if(deps[i]->data)
      delete[] deps[i]->data;
    if(deps[i]->expression)
      delete[] deps[i]->expression;
    delete deps[i];
  }
}

ArchesTable::Expr* ArchesTable::parse_addsub(string::iterator&  begin,
                                             string::iterator& end)
{
  Expr* child1 = parse_muldiv(begin, end);
  if(!child1)
    return 0;
  while(begin != end){
    char next = *begin;
    if(next == '+' || next == '-'){
      begin++;
      Expr* child2 = parse_muldiv(begin, end);
      if(!child2)
        return 0;
      child1 = new Expr(next, child1, child2);
    } else if(next == ' ' || next == '\t' || next == '\n'){
      begin++;
    } else {
      break;
    }
  }
  return child1;
}

ArchesTable::Expr* ArchesTable::parse_muldiv(string::iterator&  begin,
                                             string::iterator& end)
{
  Expr* child1 = parse_sign(begin, end);
  if(!child1)
    return 0;
  while(begin != end){
    char next = *begin;
    if(next == '*' || next == '/'){
      begin++;
      Expr* child2 = parse_sign(begin, end);
      if(!child2)
        return 0;
      child1 = new Expr(next, child1, child2);
    } else if(next == ' ' || next == '\t' || next == '\n'){
      begin++;
    } else {
      break;
    }
  }
  return child1;
}

ArchesTable::Expr* ArchesTable::parse_sign(string::iterator& begin,
                                           string::iterator& end)
{
  while(begin != end){
    char next = *begin;
    if(next == '-'){
      begin++;
      Expr* child = parse_idorconstant(begin, end);
      if(!child)
        return 0;
      return new Expr(next, child, 0);
    } else if(next == '+'){
      begin++;
      Expr* child = parse_idorconstant(begin, end);
      return child;
    } else if(next == ' ' || next == '\t' || next == '\n'){
      begin++;
    } else {
      Expr* child = parse_idorconstant(begin, end);
      return child;
    }
  }
}

ArchesTable::Expr* ArchesTable::parse_idorconstant(string::iterator& begin,
                                                   string::iterator& end)
{
  while(begin != end && (*begin == ' ' || *begin == '\t' || *begin == '\n'))
    begin++;
  if(begin == end)
    return 0;
  char next = *begin;
  if(next == '['){
    // ID...
    begin++;
    string id;
    while(begin != end && *begin != ']')
      id.push_back(*begin++);
    if(begin == end)
      return 0;
    begin++; // skip ]
    int id_index = addDependentVariable(id);
    Dep* dep = deps[id_index];
    if(dep->type == Dep::ConstantValue)
      return new Expr(dep->constantValue);
    else
      return new Expr(id_index);
  } else if(isdigit(next)){
    string constant;
    while((*begin >= '0' && *begin <= '9') || *begin == '.')
      constant.push_back(*begin++);
    if(*begin == 'e' || *begin == 'E'){
      constant.push_back(*begin++);
      if(*begin == '-' || *begin == '+')
        constant.push_back(*begin++);
      while((*begin >= '0' && *begin <= '9'))
        constant.push_back(*begin++);
    }
    istringstream in(constant);
    double c;
    in >> c;
    if(!in)
      return 0;
    return new Expr(c);
  } else {
    return 0;
  }
}

int ArchesTable::addDependentVariable(const string& name)
{
  ASSERT(!file_read);
  for(unsigned int i=0;i<deps.size();i++){
    Dep* dep = deps[i];
    if(dep->name == name)
      return i;
  }
  Dep* dep = new Dep(Dep::TableValue);
  dep->name = name;
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
    Dep* dep = new Dep(Dep::TableValue);
    
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
  for(unsigned int i=0;i<inds.size();i++)
    input_indices[i] = -1;
  for(unsigned int i=0;i<inds.size();i++){
    Ind* ind = inds[i];
    bool found = false;
    for(unsigned int j=0;j<in_inds.size();j++){
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
  for(unsigned int i=0;i<inds.size();i++){
    Ind* ind = inds[i];
    Ind* input_ind = in_inds[input_indices[i]];
    ind->name = input_ind->name;
    ind->weights = input_ind->weights;
    ind->uniform = input_ind->uniform;
    ind->dx = input_ind->dx;
    cerr << i << ", input " << input_indices[i] << ", name=" << ind-> name << ", dx=" << ind->dx << '\n';
    ind->offset.resize(ind->weights.size());
    for(unsigned int i=0;i<ind->weights.size();i++)
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
  for(unsigned int i=0;i<in_inds.size();i++){
    bool found=false;
    for(unsigned int j=0;j<inds.size();j++){
      if(in_inds[i]->name == inds[j]->name){
	found=true;
	break;
      }
    }
    if(!found){
      // Find the default value...
      bool found = false;
      double value;
      for(unsigned int j=0;j<defaults.size();j++){
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
  for(unsigned int i=0;i<deps.size();i++){
    Dep* dep = deps[i];
    if(dep->type != Dep::TableValue)
      continue;
    Dep* inputdep = 0;
    for(unsigned int j=0;j<in_deps.size();j++){
      if(in_deps[j]->name == dep->name){
	inputdep = in_deps[j];
	break;
      }
    }
    if(!inputdep)
      throw InternalError(string("Dependent variable: ")+dep->name+" not found");
    cerr << "Downslicing: " << dep->name << '\n';
    dep->data = new double[newsize];

    for(unsigned int i=0;i<inds.size();i++)
      n[i]=0;

    for(int i=0;i<newsize;i++){
      double sum = 0;
      long iidx = 0;
      for(unsigned int j=0;j<inds.size();j++){
	Ind* ind = in_inds[input_indices[j]];
	iidx += ind->offset[n[j]];
      }

      for(int j=0;j<interp_size;j++){
	long index = iidx+idx[j];
        sum += w[j]*inputdep->data[index];
      }

      long oidx = 0;
      for(unsigned int j=0;j<inds.size();j++){
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
  for(unsigned int i=0;i<deps.size();i++){
    Dep* dep = deps[i];
    if(dep->type != Dep::DerivedValue)
      continue;

    cerr << "Evaluating: " << dep->name << '\n';
    dep->data = new double[newsize];
    evaluate(dep->expression, dep->data, newsize);
  }
  file_read = true;
  double dt = Time::currentSeconds()-start;
  cerr << "Read and interpolated table in " << dt << " seconds\n";
}

void ArchesTable::evaluate(Expr* expr, double* data, int size)
{
  switch(expr->op){
  case '+':
    {
      double* data1 = new double[size];
      double* data2 = new double[size];
      evaluate(expr->child1, data1, size);
      evaluate(expr->child2, data2, size);
      for(int i=0;i<size;i++)
        data[i] = data1[i] + data2[i];
      delete[] data1;
      delete[] data2;
    }
    break;
  case '-':
    {
      double* data1 = new double[size];
      double* data2 = new double[size];
      evaluate(expr->child1, data1, size);
      evaluate(expr->child2, data2, size);
      for(int i=0;i<size;i++)
        data[i] = data1[i] - data2[i];
      delete[] data1;
      delete[] data2;
    }
    break;
  case '*':
    {
      double* data1 = new double[size];
      double* data2 = new double[size];
      evaluate(expr->child1, data1, size);
      evaluate(expr->child2, data2, size);
      for(int i=0;i<size;i++)
        data[i] = data1[i] * data2[i];
      delete[] data1;
      delete[] data2;
    }
    break;
  case '/':
    {
      double* data1 = new double[size];
      double* data2 = new double[size];
      evaluate(expr->child1, data1, size);
      evaluate(expr->child2, data2, size);
      for(int i=0;i<size;i++)
        data[i] = data1[i] / data2[i];
      delete[] data1;
      delete[] data2;
    }
    break;
  case 'c':
    {
      for(int i=0;i<size;i++)
        data[i] = expr->constant;
    }
    break;
  case 'i':
    {
      double* from = deps[expr->dep]->data;
      for(int i=0;i<size;i++)
        data[i] = from[i];
    }
    break;
  default:
    throw InternalError("Bad op in expression");
  }
}
    
void ArchesTable::interpolate(int index, CCVariable<double>& result,
			      const CellIterator& in_iter,
			      vector<constCCVariable<double> >& independents)
{
  Dep* dep = deps[index];
  switch(dep->type){
  case Dep::ConstantValue:
    {
      double value = dep->constantValue;
      for(CellIterator iter = in_iter; !iter.done(); iter++)
        result[*iter] = value;
    }
    break;
  case Dep::TableValue:
  case Dep::DerivedValue:
    {
      int ni = inds.size();
      ASSERT(ni < MAXINDEPENDENTS);
      double w[MAXINDEPENDENTS];
      long idx0[MAXINDEPENDENTS];
      long idx1[MAXINDEPENDENTS];
      long ninterp = 1<<ni;

      for(CellIterator iter = in_iter; !iter.done(); iter++){
        for(int i=0;i<ni;i++){
          Ind* ind = inds[i];
          double value = independents[i][*iter];
          if(ind->uniform){
            double index = (value-ind->weights[0])/ind->dx;
            int idx = (int)index;
            if(index < 0 || index >= ind->weights.size()){
              if(value == ind->weights[ind->weights.size()-1]){
                idx--;
              } else if(index < 0 || index > -1.e-10){
                index=0;
                idx=0;
              } else {
                cerr.precision(17);
                cerr << "value=" << value << ", start=" << ind->weights[0] << ", dx=" << ind->dx << '\n';
                cerr << "index=" << index << ", fraction=" << index-idx << '\n';
                cerr << "last value=" << ind->weights[ind->weights.size()-1] << '\n';
                throw InternalError("Interpolate outside range of table");
              }
            }
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
              weight *= w[j];
            } else {
              index += idx0[j];
              weight *= 1-w[j];
            }
          }
          double value = dep->data[index] * weight;
          sum += value;
        }
        result[*iter] = sum;
      }
    }
  }
}

double ArchesTable::interpolate(int index, vector<double>& independents)
{
  Dep* dep = deps[index];
  int ni = inds.size();
  ASSERT(ni < MAXINDEPENDENTS);
  double w[MAXINDEPENDENTS];
  long idx0[MAXINDEPENDENTS];
  long idx1[MAXINDEPENDENTS];
  long ninterp = 1<<ni;

  for(int i=0;i<ni;i++){
    Ind* ind = inds[i];
    double value = independents[i];
    if(ind->uniform){
      double index = (value-ind->weights[0])/ind->dx;
      if(index < 0 || index >= ind->weights.size()){
        cerr << "value=" << value << ", start=" << ind->weights[0] << ", dx=" << ind->dx << '\n';
        cerr << "index=" << index << '\n';
        throw InternalError("Interpolate outside range of table");
      }
      int idx = (int)index;
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
        weight *= w[j];
      } else {
        index += idx0[j];
        weight *= 1-w[j];
      }
    }
    double value = dep->data[index] * weight;
    sum += value;
  }
  return sum;
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

