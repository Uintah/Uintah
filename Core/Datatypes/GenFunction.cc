/*
  GenFunction.cc: implementation of GenFunction class

  Created by:
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    August 2000
    
    Copyright (C) 2000 SCI Group
*/


#include <Core/Datatypes/GenFunction.h>
#include <Core/Math/MinMax.h>
#include <sstream>

namespace SCIRun {

//////////
// PIO Support
static Persistent* maker(){
  return new GenFunction();
}

PersistentTypeID GenFunction::type_id("GenFunction", 
				      "Datatype", 
				      maker);

#define GENFUNCTION_VERSION 1
void GenFunction::io(Piostream& stream){

  stream.begin_class("GenFunction", GENFUNCTION_VERSION);
  
  Pio(stream, d_domDims);
  Pio(stream, d_resDims);
  Pio(stream, d_fnsDims);
  Pio(stream, d_isFast);
  Pio(stream, d_isChanged);

  Pio(stream, d_fstr);

  if(stream.reading()){
    set_funcs(d_fstr, d_domDims);
    refresh_fast_eval();
  }

  stream.end_class();
}

void GenFunction::clear(FuncPtrV& ff){
  for (unsigned int i=0; i<ff.size(); i++){
    delete ff[i];
    ff[i]=NULL;
  }
}

GenFunction::GenFunction()
  : d_fns(0), d_fevals(0), d_fstr(0), d_domDims(0), d_resDims(0), d_fnsDims(0)
{   
}
   
GenFunction::GenFunction(const GenFunction& copy) : 
  d_fevals(copy.d_fevals),
  d_fstr(copy.d_fstr),
  d_domDims(copy.d_domDims),
  d_resDims(copy.d_resDims),
  d_fnsDims(copy.d_fnsDims),
  d_isFast(copy.d_isFast),
  d_isChanged(copy.d_isChanged)
{
  d_fns.resize(d_resDims);
  for (int i=0; i<d_resDims; i++)
    d_fns[i]=new Function(copy.d_fns[i]);
  delete d_vars2;
  d_vars2=new double[d_domDims];
}
  
GenFunction::~GenFunction(){
  clear(d_fns);
  delete d_vars2;
} 
  
GenFunction& GenFunction::operator=(const GenFunction& copy){
  d_domDims=copy.d_domDims;
  d_resDims=copy.d_resDims;
  d_isFast=copy.d_isFast;
  d_isChanged=copy.d_isChanged;
  d_fstr=copy.d_fstr;
  d_fevals=copy.d_fevals;
  d_fns.resize(d_resDims);
  for (int i=0; i<d_resDims; i++)
    d_fns[i]=new Function(copy.d_fns[i]);

  delete d_vars2;
  d_vars2=new double[d_domDims];
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

  d_fnsDims=fdims;
  d_resDims=dims;
    
  // getting rid of old functions
  clear(d_fns);
    
  d_fns.resize(d_resDims);
  d_fstr.resize(d_resDims);
  d_fevals.resize(d_resDims);
  for (int i=0; i<d_resDims; i++){
    d_fns[i]=ff[i];
    d_fstr[i]=ss[i];
  }
  d_domDims=dd;
  d_isChanged=1;
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
   
  d_resDims=dims;
  d_fnsDims=fdims;
    
  // getting rid of old functions
  clear(d_fns);

  d_fns.resize(d_resDims);
  d_fstr.resize(d_resDims);
  d_fevals.resize(d_resDims);
  for (unsigned int i=0; i<ff.size(); i++){
    ostringstream ostr;
    d_fns[i]=new Function (ff[i]);
    ostr << d_fns[i];
    d_fstr[i]=ostr.str();
  }
  d_domDims=dd;
  d_isChanged=1;
  refresh_fast_eval();
}

void GenFunction::set_speed(bool is_f){
  if (is_f ^ d_isFast){
    refresh_fast_eval();
    d_isFast=is_f;
  }
}

void GenFunction::refresh_fast_eval(){
  if (d_isChanged){
    d_fevals.resize(d_fns.size());
      
    for (int i=0; i<d_resDims; i++){
      ASSERT(d_fns[i]!=NULL);
      d_fevals[i]=d_fns[i]->getFastEval();
    }
  }
}
  
string GenFunction::get_str_f(unsigned int n) const{
  if (n<d_fstr.size() && n>=0)
    return d_fstr[n];
  else
    return "0";
}
  
void GenFunction::set_num_comp(int nc){    
  ASSERT(nc>=0);
  unsigned int unc = nc;
  if (unc>=d_fns.size()){
    throw DimensionMismatch(unc, (long)d_fns.size());
  }
  d_resDims=nc+1;
}

} // End namespace SCIRun

