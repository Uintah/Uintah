/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

#if 0  
  // Commented out to fix compilation problem on linux - Steve
  Pio(stream, domDims_);
  Pio(stream, resDims_);
  Pio(stream, fnsDims_);
  Pio(stream, isFast_);
  Pio(stream, isChanged_);

  Pio(stream, fstr_);
#endif

  if(stream.reading()){
    set_funcs(fstr_, domDims_);
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
  : fns_(0), fevals_(0), fstr_(0), domDims_(0), resDims_(0), fnsDims_(0)
{   
}
   
GenFunction::GenFunction(const GenFunction& copy) : 
  fevals_(copy.fevals_),
  fstr_(copy.fstr_),
  domDims_(copy.domDims_),
  resDims_(copy.resDims_),
  fnsDims_(copy.fnsDims_),
  isFast_(copy.isFast_),
  isChanged_(copy.isChanged_)
{
  fns_.resize(resDims_);
  for (int i=0; i<resDims_; i++)
    fns_[i]=new Function(copy.fns_[i]);
  delete vars2_;
  vars2_=new double[domDims_];
}
  
GenFunction::~GenFunction(){
  clear(fns_);
  delete vars2_;
} 
  
GenFunction& GenFunction::operator=(const GenFunction& copy){
  domDims_=copy.domDims_;
  resDims_=copy.resDims_;
  isFast_=copy.isFast_;
  isChanged_=copy.isChanged_;
  fstr_=copy.fstr_;
  fevals_=copy.fevals_;
  fns_.resize(resDims_);
  for (int i=0; i<resDims_; i++)
    fns_[i]=new Function(copy.fns_[i]);

  delete vars2_;
  vars2_=new double[domDims_];
  return *this;
}

void GenFunction::set_funcs(const StrV& ss, int dd){
     
  unsigned int dims=(unsigned int)ss.size();
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

  fnsDims_=fdims;
  resDims_=dims;
    
  // getting rid of old functions
  clear(fns_);
    
  fns_.resize(resDims_);
  fstr_.resize(resDims_);
  fevals_.resize(resDims_);
  for (int i=0; i<resDims_; i++){
    fns_[i]=ff[i];
    fstr_[i]=ss[i];
  }
  domDims_=dd;
  isChanged_=1;
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
   
  resDims_=dims;
  fnsDims_=fdims;
    
  // getting rid of old functions
  clear(fns_);

  fns_.resize(resDims_);
  fstr_.resize(resDims_);
  fevals_.resize(resDims_);
  for (unsigned int i=0; i<ff.size(); i++){
    ostringstream ostr;
    fns_[i]=new Function (ff[i]);
    ostr << fns_[i];
    fstr_[i]=ostr.str();
  }
  domDims_=dd;
  isChanged_=1;
  refresh_fast_eval();
}

void GenFunction::set_speed(bool is_f){
  if (is_f ^ isFast_){
    isFast_=is_f;
    refresh_fast_eval();
  }
}

void GenFunction::refresh_fast_eval(){
  if (isChanged_ && isFast_) {
    fevals_.resize(fns_.size());
      
    for (int i=0; i<resDims_; i++){
      ASSERT(fns_[i]!=NULL);
      fevals_[i]=fns_[i]->getFastEval();
    }
  }
}
  
string GenFunction::get_str_f(unsigned int n) const
{
  if (n < fstr_.size())
    return fstr_[n];
  else
    return "0";
}
  
void GenFunction::set_num_comp(int nc){    
  ASSERT(nc>=0);
  unsigned int unc = nc;
  if (unc>=fns_.size()){
    throw DimensionMismatch((long)unc, (long)fns_.size());
  }
  resDims_=nc+1;
}

} // End namespace SCIRun

