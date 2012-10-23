/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <CCA/Components/Models/FluidsBased/ArchesTable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/IO/UintahZlibUtil.h>

#include <Core/Math/MiscMath.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <fstream>

#define MAXINDEPENDENTS 100

// TODO:  Interpolation could be a lot faster
// TODO: parentheses in expressions, other ops in expressions

using namespace std;
using namespace Uintah;

static DebugStream cerr_dbg("ARCHES_TABLE", true);

ArchesTable::ArchesTable(ProblemSpecP& params)
{
  file_read_ = false;
  params->require("filename", filename_);

  // Parse default values
  for (ProblemSpecP child = params->findBlock("defaultValue"); child != 0;
       child = child->findNextBlock("defaultValue")) {
    DefaultValue* df = scinew DefaultValue;
    if(!child->getAttribute("name", df->name))
      throw ProblemSetupException("No name for defaultValue", __FILE__, __LINE__);
    child->get(df->value);
    defaults.push_back(df);
  }

  // Parse constant values
  for (ProblemSpecP child = params->findBlock("constantValue"); child != 0;
       child = child->findNextBlock("constantValue")) {
    Dep* dep = scinew Dep(Dep::ConstantValue);
    if(!child->getAttribute("name", dep->name))
      throw ProblemSetupException("No name for constantValue", __FILE__, __LINE__);
    child->get(dep->constantValue);
    deps.push_back(dep);
  }

  // Parse derived values
  for (ProblemSpecP child = params->findBlock("derivedValue"); child != 0;
       child = child->findNextBlock("derivedValue")) {
    Dep* dep = scinew Dep(Dep::DerivedValue);
    if(!child->getAttribute("name", dep->name))
      throw ProblemSetupException("No expression for derivedValue", __FILE__, __LINE__);

    child->get(dep->expr_string);
    string expr = dep->expr_string;
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
      throw ProblemSetupException("Error parsing expression", __FILE__, __LINE__);
    }
    deps.push_back(dep);
  }
}

ArchesTable::~ArchesTable()
{
  for(int i=0;i<(int)defaults.size();i++)
    delete defaults[i];
  for(int i=0;i<(int)inds.size();i++)
    delete inds[i];
  for(int i=0;i<(int)deps.size();i++){
    if(deps[i]->data)
      delete[] deps[i]->data;
    if(deps[i]->expression)
      delete deps[i]->expression;
    delete deps[i];
  }
}

void
ArchesTable::outputProblemSpec(ProblemSpecP& ps)
{
  ps->setAttribute("type", "Arches");
  ps->appendElement("filename",filename_);
  for (vector<DefaultValue*>::const_iterator it = defaults.begin();
       it != defaults.end(); ++it)
    (*it)->outputProblemSpec(ps);

  for (vector<Dep*>::const_iterator it = deps.begin(); it != deps.end(); ++it)
    (*it)->outputProblemSpec(ps);
}

ArchesTable::Expr*
ArchesTable::parse_addsub( string::iterator & begin,
                           string::iterator & end )
{
  Expr* child1 = parse_muldiv(begin, end);
  if( !child1 ) {
    return 0;
  }
  while(begin != end){
    char next = *begin;
    if(next == '+' || next == '-'){
      begin++;
      Expr* child2 = parse_muldiv(begin, end);
      if(!child2)
        return 0;
      child1 = scinew Expr(next, child1, child2);
    } else if(next == ' ' || next == '\t' || next == '\n'){
      begin++;
    } else {
      break;
    }
  }
  return child1;
}

ArchesTable::Expr*
ArchesTable::parse_muldiv( string::iterator& begin,
                           string::iterator& end )
{
  Expr* child1 = parse_sign(begin, end);
  if( !child1 ) {
    return 0;
  }
  while(begin != end) {
    char next = *begin;
    if(next == '*' || next == '/'){
      begin++;
      Expr* child2 = parse_sign(begin, end);
      if(!child2)
        return 0;
      child1 = scinew Expr(next, child1, child2);
    } else if(next == ' ' || next == '\t' || next == '\n'){
      begin++;
    } else {
      break;
    }
  }
  return child1;
}

ArchesTable::Expr*
ArchesTable::parse_sign( string::iterator& begin,
                         string::iterator& end )
{
  while(begin != end) {
    char next = *begin;
    if(next == '-') {
      begin++;
      Expr* child = parse_idorconstant(begin, end);
      if(!child)
        return 0;
      return scinew Expr(next, child, 0);
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
  return 0;
}

ArchesTable::Expr*
ArchesTable::parse_idorconstant( string::iterator& begin,
                                 string::iterator& end )
{
  while(begin != end && (*begin == ' ' || *begin == '\t' || *begin == '\n')) {
    begin++;
  }
  if(begin == end) {
    return 0;
  }
  char next = *begin;
  if(next == '[') {
    // ID...
    begin++;
    string id;
    while(begin != end && *begin != ']') {
      id.push_back(*begin++);
    }
    if(begin == end) {
      return 0;
    }
    begin++; // skip ]

    int id_index = addDependentVariable(id);
    Dep* dep = deps[id_index];
    if(dep->type == Dep::ConstantValue)
      return scinew Expr(dep->constantValue);
    else
      return scinew Expr(id_index);
  } else if(next == '{') {
    // Independent variable...
    begin++;
    string id;
    while(begin != end && *begin != '}') {
      id.push_back(*begin++);
    }
    if(begin == end) {
      return 0;
    }
    begin++; // skip }

    return scinew Expr(id);
  } else if(next =='(') {
    // Parenthetical
    begin++;
    Expr* child = parse_addsub(begin, end);
    if(!child || *begin++ != ')')
      return 0;
    else
      return child;
  } else if(isdigit(next)) {
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
    return scinew Expr(c);
  } else {
    return 0;
  }
}

int
ArchesTable::addDependentVariable( const string & name )
{
  ASSERT(!file_read_);
  for(int i=0;i<static_cast<int>(deps.size());i++) {
    Dep* dep = deps[i];
    if(dep->name == name)
      return i;
  }
  Dep* dep = scinew Dep(Dep::TableValue);
  dep->name = name;
  deps.push_back(dep);
  return (int)deps.size()-1;
}

void
ArchesTable::addIndependentVariable( const string& name )
{
  ASSERT(!file_read_);
  Ind* ind = scinew Ind;
  ind->name = name;
  inds.push_back(ind);
}

void
ArchesTable::setup(const bool cerrSwitch)
{
  cerr_dbg.setActive(cerrSwitch);
  double start = Time::currentSeconds();
  // Read the index...


  gzFile gzFp = gzopen( filename_.c_str(), "r" );

  if( gzFp == NULL ) {
    // If errno is 0, then not enough memory to uncompress file.
    cout << "Error: gz open failed for file: '" << filename_ << "'.  (Errno: " << errno << ")\n";
    throw ProblemSetupException("Unable to open the given input file: " + filename_, __FILE__, __LINE__);
  }

  cerr_dbg << "Reading reaction table: " << filename_ << endl;
  int nvars = getInt( gzFp );
  cerr_dbg << "Reading " << nvars << " variables : ";

  vector<Ind*> in_inds(nvars);
  vector<int> axis_sizes(nvars);

  // Read the names.
  for( int i = 0; i < nvars; i++ ) {
    Ind* ind = scinew Ind;
    ind->name = getString( gzFp );
    in_inds[i] = ind;
  }

  // Read the size of each axis
  for( int i = 0; i < nvars; i++ ) {
    int num = getInt( gzFp );
    axis_sizes[i] = num;
    if(num <= 2)
      throw InternalError("Table must have at least size 2 in each dimension", __FILE__, __LINE__);
  }

  for(int i=0;i<nvars;i++) {
    cerr_dbg << axis_sizes[i] << " ";
  }
  cerr_dbg << '\n';

  // Set up the axes.
  // The first variable may have different weights for each dependent
  // variable
  vector<InterpAxis*> in_axes(nvars);
  long stride = axis_sizes[0];
  in_axes[0] = 0;
  for( int i = nvars-1; i >= 1; i-- ) {
    in_axes[i] = scinew InterpAxis(axis_sizes[i], stride);
    in_axes[i]->useCount++;
    stride *= axis_sizes[i];
  }
  long size = stride;

  int ndeps = getInt( gzFp );
  vector<Dep*> in_deps(ndeps);
  for(int j=0;j<ndeps;j++) {
    Dep* dep = scinew Dep(Dep::TableValue);
    
    dep->name = getString( gzFp );
    dep->data = scinew double[size];
    // Add the first (typically masss fraction) axis with stride 1
    dep->addAxis(scinew InterpAxis(axis_sizes[0], 1));
    for(int i=1;i<nvars;i++)
      dep->addAxis(in_axes[i]);
    in_deps[j] = dep;
  }

  // Next, read units - they may not all be there, so we just read the
  // line and throw it away...
  getLine( gzFp );

  // Next, read the axis weights for everything but the first axis
  for(int i=1;i<nvars;i++) {
    InterpAxis* axis = in_axes[i];
    int n=axis_sizes[i];
    for(int i=0;i<n;i++)
      axis->weights[i] = getDouble( gzFp );
  }
  int sizefirst = axis_sizes[0];
  for(int idep=0;idep<ndeps;idep++) {
    Dep* dep = in_deps[idep];
    for(int i=0;i<sizefirst;i++) {
      // Read axis weights
      double weight = getDouble( gzFp );
      dep->axes[0]->weights[i] = weight;
    }
    for(int i=0;i<size;i++) {
      // Read values for this dependent variable
      double value = getDouble( gzFp );
      dep->data[i] = value;
    }
  }

  // At this point, the entire file should have been parsed... so if
  // there is anything left in the file, it is an error... let the
  // user know in this case.
  //
  string data = getString( gzFp );
  if( data != "" ) {
    throw InternalError( "Extra data found in file after parsing... (starting with " + 
                         data + ".\n", __FILE__, __LINE__ );
  }

  // finalize axes
  for(int i=1;i<nvars;i++)
    in_axes[i]->finalize();
  for(int i=0;i<ndeps;i++)
    in_deps[i]->axes[0]->finalize();

  // Map the desired variables to the input variables
  vector<int> axis_map(inds.size(), -1);
  for(int i=0;i<static_cast<int>(inds.size());i++) {
    Ind* ind = inds[i];
    // Look in the alias map
    bool found = false;
    for(int j=0;j<static_cast<int>(in_inds.size());j++) {
      if(in_inds[j]->name == ind->name){
        found=true;
        axis_map[i] = j;
        break;
      }
    }
    if(!found)
      throw InternalError(string("Independent variable: ")+ind->name+" not found", __FILE__, __LINE__);
  }

  // Create the new axes
  vector<InterpAxis*> new_axes(inds.size());
  long newstride = 1;
  long firststride = 0;
  for(int i=0;i<static_cast<int>(inds.size());i++){
    if(axis_map[i] == 0){
      // This is the first "special" axis - treat it specially
      firststride = newstride;
      new_axes[i] = 0;
    } else {
      InterpAxis* in_axis = in_axes[axis_map[i]];
      new_axes[i] = scinew InterpAxis(in_axis, newstride);
    }
    newstride *= axis_sizes[axis_map[i]];
  }
  ASSERT(firststride != 0);
  long newsize = newstride;

  // Down-slice the table if necessary
  int dim_diff = in_inds.size()-inds.size();
  long interp_size = 1<<dim_diff;
  long* idx = scinew long[interp_size];
  double* w =scinew double[interp_size];

  for(int idep=0;idep<static_cast<int>(deps.size());idep++){
    Dep* dep = deps[idep];
    if(dep->type != Dep::TableValue)
      continue;
    Dep* inputdep = 0;
    for(int j=0;j<static_cast<int>(in_deps.size());j++){
      if(in_deps[j]->name == dep->name){
        inputdep = in_deps[j];
        break;
      }
    }
    if(!inputdep) {
      throw InternalError(string("Dependent variable: ")+dep->name+" not found", __FILE__, __LINE__);
    }
    cerr_dbg << "Downslicing: " << dep->name << '\n';
    dep->data =scinew double[newsize];

    // Build the axes
    for(int i=0;i<static_cast<int>(inds.size());i++){
      if(new_axes[i]) {
        // Use as is
        dep->addAxis(new_axes[i]);
      } else {
        // Need a new axis here. Look at all of the previous deps
        // and see if we can find an axis that we can rob
        InterpAxis* newAxis = 0;
        for(int d=0;d<idep-1;d++){
          if(deps[d]->type == Dep::TableValue){
            InterpAxis* a = deps[d]->axes[i];
            if(a->sameAs(inputdep->axes[axis_map[i]])){
              newAxis = a;
              break;
            }
          }
        }
        if(!newAxis)
          newAxis = scinew InterpAxis(inputdep->axes[axis_map[i]],
                                      firststride);
        dep->addAxis(newAxis);
      }
    }
  
    // Find the axes to be eliminated...
    for(int j=0;j<interp_size;j++){
      idx[j] = 0;
      w[j] = 1;
    }
    long s = 1;
    for(int i=0;i<static_cast<int>(in_inds.size());i++){
      bool found=false;
      for(int j=0;j<static_cast<int>(inds.size());j++){
        if(in_inds[i]->name == inds[j]->name){
          found=true;
          break;
        }
      }
      if(found)
        continue;

      // Find the default value...
      found = false;
      double value = 0;
      for(int j=0;j<static_cast<int>(defaults.size());j++){
        if(in_inds[i]->name == defaults[j]->name){
          found=true;
          value=defaults[j]->value;
        }
      }
      if(!found)
        throw InternalError("Default value for "+in_inds[i]->name+" not found", __FILE__, __LINE__);
      InterpAxis* axis = inputdep->axes[i];
      int l=0;
      int h=axis->weights.size()-1;
      if(value < axis->weights[l] || value > axis->weights[h])
        throw InternalError("Interpolate outside range of table", __FILE__, __LINE__);
      while(h > l+1){
        int m = (h+l)/2;
        if(value < axis->weights[m])
          h=m;
        else
          l=m;
      }
      long i0 = axis->offset[l];
      long i1 = axis->offset[h];
      double w0 = (value-axis->weights[l])/(axis->weights[h]-axis->weights[l]);
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
      s<<=1;
    }

    // Interpolate...
    vector<int> n(inds.size(), 0);
    for(int i=0;i<newsize;i++){
      double sum = 0;
      long iidx = 0;
      // Determine the source indices...
      for(int j=0;j<static_cast<int>(inds.size());j++){
        InterpAxis* in_axis = inputdep->axes[axis_map[j]];
        iidx += in_axis->offset[n[j]];
      }

      // Interpolate
      for(int j=0;j<interp_size;j++){
        long index = iidx+idx[j];
        sum += w[j]*inputdep->data[index];
      }

      // Determine the output index
      long oidx = 0;
      for(int j=0;j<static_cast<int>(inds.size());j++){
        oidx += dep->axes[j]->offset[n[j]];
      }
      dep->data[oidx] = sum;
      int s = inds.size()-1;
      while(s >= 0 && ++n[s] >= (int)dep->axes[s]->weights.size()){
        n[s]=0;
        s--;
      }
      
    }
  }

  // Free up the input deps
  delete idx;
  delete w;
  for(int i=0;i<(int)in_inds.size();i++)
    delete in_inds[i];
  for(int i=1;i<(int)in_axes.size();i++)
    if(--in_axes[i]->useCount == 0)
      delete in_axes[i];
  for(int i=0;i<(int)in_deps.size();i++){
    if(in_deps[i]->data)
      delete[] in_deps[i]->data;
    if(in_deps[i]->expression)
      delete in_deps[i]->expression;
    delete in_deps[i];
  }  
  
  // Compute derived values
  for(int i=0;i<static_cast<int>(deps.size());i++){
    Dep* dep = deps[i];
    if(dep->type != Dep::DerivedValue)
      continue;

    cerr_dbg << "Evaluating: " << dep->name << '\n';
    dep->data =scinew double[newsize];
    vector<InterpAxis*> axes;
    evaluate(dep->expression, axes, dep->data, newsize);
    for(int i=0;i<static_cast<int>(axes.size());i++)
      dep->addAxis(axes[i]);
  }
  file_read_ = true;
  double dt = Time::currentSeconds()-start;
  cerr_dbg << "Read and interpolated table in " << dt << " seconds\n";
}

void
ArchesTable::checkAxes( const vector<InterpAxis*> & a,
                        const vector<InterpAxis*> & b,
                        vector<InterpAxis*> &       out_axes)
{
  // If either of the axes are empty, use the ohter one...
  if(a.size() == 0) {
    if(b.size() != 0)
      out_axes=b;
    return;
  } else {
    if(b.size() == 0) {
      out_axes=a;
      return;
    }
  }
  if(a.size() != b.size()) {
    throw InternalError("Cannot compute a derived quantity on variables with different dimension", 
                        __FILE__, __LINE__);
  }
  for(int i=0;i<static_cast<int>(a.size());i++) {
    if(!a[i]->sameAs(b[i]))
      throw InternalError("Cannot compute a derived quantity on variables with different axes",
                          __FILE__, __LINE__);
  }
  // They are the same, okay...
  out_axes=a;
}

void
ArchesTable::evaluate(Expr* expr, vector<InterpAxis*>& out_axes,
                      double* data, int size)
{
  switch(expr->op) {
  case '+':
    {
      vector<InterpAxis*> axes1;
      vector<InterpAxis*> axes2;
      double* data1 =scinew double[size];
      double* data2 =scinew double[size];
      evaluate(expr->child1, axes1, data1, size);
      evaluate(expr->child2, axes2, data2, size);
      checkAxes(axes1, axes2, out_axes);
      for(int i=0;i<size;i++)
        data[i] = data1[i] + data2[i];
      delete[] data1;
      delete[] data2;
    }
    break;
  case '-':
    {
      vector<InterpAxis*> axes1;
      vector<InterpAxis*> axes2;
      double* data1 =scinew double[size];
      double* data2 =scinew double[size];
      evaluate(expr->child1, axes1, data1, size);
      evaluate(expr->child2, axes2, data2, size);
      checkAxes(axes1, axes2, out_axes);
      for(int i=0;i<size;i++)
        data[i] = data1[i] - data2[i];
      delete[] data1;
      delete[] data2;
    }
    break;
  case '*':
    {
      vector<InterpAxis*> axes1;
      vector<InterpAxis*> axes2;
      double* data1 =scinew double[size];
      double* data2 =scinew double[size];
      evaluate(expr->child1, axes1, data1, size);
      evaluate(expr->child2, axes2, data2, size);
      checkAxes(axes1, axes2, out_axes);
      for(int i=0;i<size;i++)
        data[i] = data1[i] * data2[i];
      delete[] data1;
      delete[] data2;
    }
    break;
  case '/':
    {
      vector<InterpAxis*> axes1;
      vector<InterpAxis*> axes2;
      double* data1 =scinew double[size];
      double* data2 =scinew double[size];
      evaluate(expr->child1, axes1, data1, size);
      evaluate(expr->child2, axes2, data2, size);
      checkAxes(axes1, axes2, out_axes);
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
      int idx;
      for(idx=0;idx<static_cast<int>(inds.size());idx++){
        if(inds[idx]->name == expr->id)
          break;
      }
      if(idx == static_cast<int>(inds.size()))
        throw InternalError("Cannot find variable in expression: "+expr->id, __FILE__, __LINE__);
      int firstdep=0;
      while(firstdep != static_cast<int>(deps.size())
            && static_cast<int>(deps[firstdep]->axes.size()) == 0)
        firstdep++;
      if(firstdep == static_cast<int>(deps.size()))
        throw InternalError("Cannot find a variable with axes", __FILE__, __LINE__);
      out_axes=deps[firstdep]->axes;
      InterpAxis* axis = out_axes[idx];
      int stride = axis->offset[1]-axis->offset[0];
      int n = axis->weights.size();
      int skip = n*stride;
      int index=0;
      for(int i=0;i<size;i+=skip){
        for(int j=0;j<n;j++){
          double value = axis->weights[j];
          for(int k=0;k<stride;k++){
              data[index++]=value;
          }
        }
      }
    }
    break;
  case 'd':
    {
      double* from = deps[expr->var]->data;
      for(int i=0;i<size;i++)
        data[i] = from[i];
      out_axes = deps[expr->var]->axes;
    }
    break;
  default:
    throw InternalError("Bad op in expression", __FILE__, __LINE__);
  }
}
    
void
ArchesTable::interpolate( int index, CCVariable<double>& result,
                          const CellIterator& in_iter,
                          vector<constCCVariable<double> >& independents )
{
  Dep* dep = deps[index];
  switch(dep->type) {
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
      int ni = dep->axes.size();
      ASSERT(ni < MAXINDEPENDENTS);
      double w[MAXINDEPENDENTS];
      long idx0[MAXINDEPENDENTS];
      long idx1[MAXINDEPENDENTS];
      long ninterp = 1<<ni;

      for(CellIterator iter = in_iter; !iter.done(); iter++){
        for(int i=0;i<ni;i++){
          InterpAxis* axis = dep->axes[i];
          double value = independents[i][*iter];
          if(axis->uniform){
            double index = (value-axis->weights[0])/axis->dx;
            int idx = (int)index;
            if(index < 0 || index >= axis->weights.size()-1){
              if(value == axis->weights[axis->weights.size()-1]){
                idx--;
              } else if(index < 0 || index > -1.e-10){
                index=0;
                idx=0;
              } else {
                cerr.precision(17);
                cerr << "value=" << value << ", start=" << axis->weights[0] << ", dx=" << axis->dx << '\n';
                cerr << "index=" << index << ", fraction=" << index-idx << '\n';
                cerr << "last value=" << axis->weights[axis->weights.size()-1] << '\n';
                throw InternalError("Interpolate outside range of table", __FILE__, __LINE__);
              }
            }
            w[i] = index-idx;
            idx0[i] = axis->offset[idx];
            idx1[i] = axis->offset[idx+1];
          } else {
            int l=0;
            int h=axis->weights.size()-1;
            if(value < axis->weights[l] || value > axis->weights[h]){
              if(value < axis->weights[l] && value > axis->weights[l]-1.5e-1)
                value = axis->weights[l];
              else if(value > axis->weights[h] && value < axis->weights[h]+1.5e-1)
                value = axis->weights[h];
              else {
                cerr.precision(17);
                cerr << *iter << ", value=" << value << ", low=" << axis->weights[l] << ", high=" << axis->weights[h] << "\n";
                throw InternalError("Interpolate outside range of table", __FILE__, __LINE__);
              }
            }
            while(h > l+1){
              int m = (h+l)/2;
              if(value < axis->weights[m])
                h=m;
              else
                l=m;
            }
            idx0[i] = axis->offset[l];
            idx1[i] = axis->offset[h];
            w[i] = (value-axis->weights[l])/(axis->weights[h]-axis->weights[l]);
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

double
ArchesTable::interpolate( int index, vector<double>& independents )
{
  Dep* dep = deps[index];
  int ni = dep->axes.size();
  if(dep->type == Dep::ConstantValue) {
    return dep->constantValue;
  }
  ASSERT(ni < MAXINDEPENDENTS);
  ASSERT(ni == static_cast<int>(independents.size()));

  double w[MAXINDEPENDENTS];
  long idx0[MAXINDEPENDENTS];
  long idx1[MAXINDEPENDENTS];
  long ninterp = 1<<ni;

  for(int i=0;i<ni;i++){
    InterpAxis* axis = dep->axes[i];
    double value = independents[i];
    if(axis->uniform){
      double index = (value-axis->weights[0])/axis->dx;
      if(index < 0 || index >= axis->weights.size()){
        cerr << "value=" << value << ", start=" << axis->weights[0] << ", dx=" << axis->dx << '\n';
        cerr << "index=" << index << '\n';
        throw InternalError("Interpolate outside range of table", __FILE__, __LINE__);
      }
      int idx = (int)index;
      w[i] = index-idx;
      idx0[i] = axis->offset[idx];
      idx1[i] = axis->offset[idx+1];
    } else {
      int l=0;
      int h=axis->weights.size()-1;
      if(value < axis->weights[l] || value > axis->weights[h]){
        if(value < axis->weights[l] && value > axis->weights[l]-1.e-1)
          value = axis->weights[l];
        else if(value > axis->weights[h] && value < axis->weights[h]+1.e-1)
          value = axis->weights[h];
        else {
          cerr.precision(17);
          cerr << "value=" << value << ", low=" << axis->weights[l] << ", high=" << axis->weights[h] << "\n";
          throw InternalError("Interpolate outside range of table", __FILE__, __LINE__);
        }
      }
      while(h > l+1){
        int m = (h+l)/2;
        if(value < axis->weights[m])
          h=m;
        else
          l=m;
      }
      idx0[i] = axis->offset[l];
      idx1[i] = axis->offset[h];
      w[i] = (value-axis->weights[l])/(axis->weights[h]-axis->weights[l]);
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

#if 0
void
ArchesTable::error(istream& in)
{
  string s;
  getline(in, s);
  throw InternalError("Error parsing table, text follows: "+s, __FILE__, __LINE__);
}

int
ArchesTable::getInt(istream& in)
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
    if(c == ' ' || c == '\t' || c == '\n' || !in)
      break;
    result.push_back(c);
  }
  result.erase(result.find_last_not_of(" \t\n")+1);
  return result;
}

string ArchesTable::getLine(istream& in)
{
  eatWhite(in);
  if(startline)
    skipComments(in);
  eatWhite(in);

  string result;
  eatWhite(in);
  for(;;){
    int c = in.get();
    if(c == '\n' || !in)
      break;
    result.push_back(c);
  }
  result.erase(result.find_last_not_of(" \t\n")+1);
  return result;
}

void
ArchesTable::skipComments(istream& in)
{
  eatWhite(in);
  int c = in.get();
  while(in && c == '#'){
    while(in && c != '\n')
      c = in.get();
    eatWhite(in);
    c=in.get();
  }
  in.unget();
}

void
ArchesTable::eatWhite(istream& in)
{
  int c = in.get();
  while(in && isspace(c))
    c = in.get();
  if(in)
    in.unget();
}
#endif

void
ArchesTable::Dep::outputProblemSpec(ProblemSpecP& ps)
{
  if (type == Dep::ConstantValue) {
    stringstream ss;
    ss << constantValue;
    ProblemSpecP cv_ps = ps->appendElement("constantValue",ss.str());
    cv_ps->setAttribute("name",name);
  }
  if (type == Dep::DerivedValue) {
    ProblemSpecP dv_ps = ps->appendElement("derivedValue",expr_string);
    dv_ps->setAttribute("name",name);
  }
}

void
ArchesTable::Dep::addAxis(InterpAxis* newAxis)
{
  newAxis->useCount++;
  axes.push_back(newAxis);
}

ArchesTable::Dep::~Dep()
{
  for(int i=0;i<static_cast<int>(axes.size());i++)
    if(!--axes[i]->useCount)
      delete axes[i];
}

ArchesTable::InterpAxis::InterpAxis(int size, int stride)
  : weights(size), offset(size), useCount(0)
{
  for(int i=0;i<size;i++)
    offset[i] = i*stride;
}

ArchesTable::InterpAxis::InterpAxis(const InterpAxis* copy, int newStride) : 
  weights(copy->weights), offset(copy->offset.size()),
  uniform(copy->uniform), dx(copy->dx), useCount(0)
{
  int size = weights.size();
  for(int i=0;i<size;i++)
    offset[i] = i*newStride;
}

bool
ArchesTable::InterpAxis::sameAs(const InterpAxis* b) const
{
  if(this == b)
    return true;
  if(weights.size() != b->weights.size() || dx != b->dx || uniform != b->uniform)
    return false;
  for(int i=0;i<static_cast<int>(weights.size());i++)
    if(weights[i] != b->weights[i])
      return false;
  return true;
}

void
ArchesTable::InterpAxis::finalize()
{
  int n = weights.size();
  dx = (weights[n-1]-weights[0])/(n-1);
  if(dx <= 0)
    throw InternalError("Backwards table not supported", __FILE__, __LINE__);
  uniform = true;
  for(int i=1;i<n;i++){
    double dd = weights[i]-weights[i-1];
    if(dd<0){
      throw InternalError("Backwards table not supported(2)", __FILE__, __LINE__);
    }
    if(Abs(dd-dx)/dx > 1.e-5){
      uniform = false;
      dx = 0;
      break;
    }
  }
}
