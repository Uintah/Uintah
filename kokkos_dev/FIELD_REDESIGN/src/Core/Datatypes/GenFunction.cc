/*
  GenFunction.cc: implementation of GenFunction class

  Created by:
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    August 2000
    
    Copyright (C) 2000 SCI Group
*/


#include <SCICore/Datatypes/GenFunction.h>
#include <SCICore/Math/MinMax.h>
#include <sstream>

namespace SCICore{
namespace Datatypes{
 
using SCICore::Math::Max;

//////////
// Persistent representation
PersistentTypeID GenFunction::type_id("GenFunction", "Datatype", 0);

void GenFunction::io(Piostream&){
  NOT_FINISHED("GenFunction::io()");
}

void GenFunction::clear(FuncPtrV& ff){
  for (unsigned int i=0; i<ff.size(); i++){
    delete ff[i];
    ff[i]=NULL;
  }
}

GenFunction::GenFunction()
  : fns(0), fevals(0), fstr(0), dom_dims(0), res_dims(0), fns_dims(0)
{   
}
   
GenFunction::GenFunction(const GenFunction& copy) : 
  fevals(copy.fevals),
  fstr(copy.fstr),
  dom_dims(copy.dom_dims),
  res_dims(copy.res_dims),
  fns_dims(copy.fns_dims),
  is_fast(copy.is_fast),
  is_changed(copy.is_changed)
{
  fns.resize(res_dims);
  for (int i=0; i<res_dims; i++)
    fns[i]=new Function(copy.fns[i]);
  delete vars2;
  vars2=new double[dom_dims];
}
  
GenFunction::~GenFunction(){
  clear(fns);
  delete vars2;
} 
  
GenFunction& GenFunction::operator=(const GenFunction& copy){
  dom_dims=copy.dom_dims;
  res_dims=copy.res_dims;
  is_fast=copy.is_fast;
  is_changed=copy.is_changed;
  fstr=copy.fstr;
  fevals=copy.fevals;
  fns.resize(res_dims);
  for (int i=0; i<res_dims; i++)
    fns[i]=new Function(copy.fns[i]);

  delete vars2;
  vars2=new double[dom_dims];
  return *this;
}

void GenFunction::set_funcs(const StrV& ss, int dd){
     
  unsigned int dims=ss.size();
  int fdims=0;
    
  ASSERT(ss.size()>0 && ss.size()>=dims);
   
  FuncPtrV ff(dims, 0);
  try {
    for (unsigned int i=0; i<dims; i++){
      Function* tmp=0;
      fnparsestring(ss[i].c_str(), &tmp);
      if (tmp){
	ff[i]=tmp;
	int d=tmp->get_dims();
	if (d>dd)
	  throw DimensionMismatch(d, dd);
	fdims=Max(fdims, d);
      }
      else {
	throw InternalError("Cann't create function: bad string supplied");
      }
    }
  }
  catch (...){
    clear (ff);
    throw;
  }

  fns_dims=fdims;
  res_dims=dims;
    
  // getting rid of old functions
  clear(fns);
    
  fns.resize(res_dims);
  fstr.resize(res_dims);
  fevals.resize(res_dims);
  for (int i=0; i<res_dims; i++){
    fns[i]=ff[i];
    fstr[i]=ss[i];
  }
  dom_dims=dd;
  is_changed=1;
  set_speed(0);
}

void GenFunction::set_funcs(const FuncPtrV& ff, int dd){

  unsigned int dims=ff.size();
  int fdims=0;
    
  ASSERT(ff.size()>0 && ff.size()>=dims);    

  for (unsigned int i=0; i<dims; i++){
    if (ff[i]==NULL)
      throw InternalError("NULL pointer supplied");
    else {
      int d=ff[i]->get_dims();
      if (d>dd)
	throw DimensionMismatch(d, dd);
      fdims=Max(fdims, d);
    }
  }
   
  res_dims=dims;
  fns_dims=fdims;
    
  // getting rid of old functions
  clear(fns);

  fns.resize(res_dims);
  fstr.resize(res_dims);
  fevals.resize(res_dims);
  for (unsigned int i=0; i<ff.size(); i++){
    ostringstream ostr;
    fns[i]=new Function (ff[i]);
    ostr << fns[i];
    fstr[i]=ostr.str();
  }
  dom_dims=dd;
  is_changed=1;
  refresh_fast_eval();
}

void GenFunction::set_speed(bool is_f){
  if (is_f ^ is_fast){
    refresh_fast_eval();
    is_fast=is_f;
  }
}

void GenFunction::refresh_fast_eval(){
  if (is_changed){
    fevals.resize(fns.size());
      
    for (int i=0; i<res_dims; i++){
      ASSERT(fns[i]!=NULL);
      fevals[i]=fns[i]->getFastEval();
    }
  }
}
  
string GenFunction::get_str_f(unsigned int n) const{
  if (n<fstr.size() && n>=0)
    return fstr[n];
  else
    return "0";
}
  
void GenFunction::set_num_comp(int nc){    
  ASSERT(nc>=0);
  unsigned int unc = nc;
  if (unc>=fns.size()){
    throw DimensionMismatch(unc, (long)fns.size());
  }
  res_dims=nc+1;
}

} // Datatypes
} // SCICore

