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

// NrrdData.cc - Interface to Gordon's Nrrd package
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//
//  Copyright (C) 2001 SCI Institute

#include <Teem/Core/Datatypes/NrrdData.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cerr;

namespace SCITeem {

static Persistent* make_NrrdData() {
  return scinew NrrdData;
}

PersistentTypeID NrrdData::type_id("NrrdData", "Datatype", make_NrrdData);

NrrdData::NrrdData(bool owned) : 
  nrrd(nrrdNew()),
  data_owned_(owned)
{}

NrrdData::NrrdData(const NrrdData &copy) :
  nrrd_fname_(copy.nrrd_fname_) 
{
  nrrd = nrrdNew();
  nrrdCopy(nrrd, copy.nrrd);
  copy_sci_data(copy);
}

NrrdData::~NrrdData() {
  if(data_owned_) {
    nrrdNuke(nrrd);
  } else {
    nrrdNix(nrrd);
  }
}

NrrdData* 
NrrdData::clone() 
{
  return new NrrdData(*this);
}


// This needs to parse axis 0 and see if the label is tuple as well...
bool
NrrdData::is_sci_nrrd() const 
{
  return (originating_field_.get_rep() != 0);
}

void 
NrrdData::copy_sci_data(const NrrdData &cp)
{
  originating_field_ = cp.originating_field_;
}

int
NrrdData::get_tuple_axis_size() const
{
  vector<string> elems;
  get_tuple_indecies(elems);
  return elems.size();
}

bool
NrrdData::get_tuple_indecies(vector<string> &elems) const
{
  if (!nrrd) return false;
  string tup(nrrd->axis[0].label);
  
  string::size_type s, e;
  s = 0;
  e = 0;

  while (e < (tup.size() - 1)) {
    e = tup.find(",", s);
    elems.push_back(tup.substr(s, e));
    s = e + 1;
  }
  return true;
}

bool 
NrrdData::get_tuple_index_info(int tmin, int tmax, int &min, int &max) const
{
  if (!nrrd) return false;
  string tup(nrrd->axis[0].label);
  vector<string> elems;
  get_tuple_indecies(elems);

  if (tmin < 0 || tmin > (int)elems.size() - 1 || 
      tmax < 0 || tmax > (int)elems.size() - 1 ) return false;

  min = 0;
  max = 0;
  for (int i = 0; i <= tmax; i++) {
    
    string &s = elems[i];
    int inc = 0;
    if (s.find(string("Scalar")) <= tup.size() - 1) {
      inc = 1;
    } else if (s.find(string("Vector")) <= tup.size() - 1) {
      inc = 3;
    } else if (s.find(string("Tensor")) <= tup.size() - 1) {
      inc = 6;
    }
    if (tmin > i) min+=inc;
    if (tmax > i) max+=inc;
    if (tmax == i) max+= inc - 1;
    
  } 
  return true;
}

#define NRRDDATA_VERSION 2

//////////
// PIO for NrrdData objects
void NrrdData::io(Piostream& stream) {
  int version =  stream.begin_class("NrrdData", NRRDDATA_VERSION);
  if (stream.reading()) {
    Pio(stream, nrrd_fname_);
    if (nrrdLoad(nrrd = nrrdNew(), strdup(nrrd_fname_.c_str()))) {
      char *err = biffGet(NRRD);
      cerr << "Error reading nrrd " << nrrd_fname_ << ": " << err << endl;
      free(err);
      biffDone(NRRD);
      return;
    }
  } else { // writing

    // the nrrd file name will just append .nrrd
    nrrd_fname_ = stream.file_name + string(".nrrd");
    Pio(stream, nrrd_fname_);
    if (nrrdSave(strdup(nrrd_fname_.c_str()), nrrd, 0)) {
      char *err = biffGet(NRRD);      
      cerr << "Error writing nrrd " << nrrd_fname_ << ": "<< err << endl;
      free(err);
      biffDone(NRRD);
      return;
    }
  }
  if (version > 1) {
    Pio(stream, data_owned_);
    Pio(stream, originating_field_);
  }
  stream.end_class();
}


template <>
unsigned int get_nrrd_type<char>() {
  return nrrdTypeChar;
}


template <>
unsigned int get_nrrd_type<unsigned char>()
{
  return nrrdTypeUChar;
}

template <>
unsigned int get_nrrd_type<short>()
{
  return nrrdTypeShort;
}

template <>
unsigned int get_nrrd_type<unsigned short>()
{
  return nrrdTypeUShort;
}

template <>
unsigned int get_nrrd_type<int>()
{
  return nrrdTypeInt;
}

template <>
unsigned int get_nrrd_type<unsigned int>()
{
  return nrrdTypeUInt;
}

template <>
unsigned int get_nrrd_type<long long>()
{
  return nrrdTypeLLong;
}

template <>
unsigned int get_nrrd_type<unsigned long long>()
{
  return nrrdTypeULLong;
}

template <>
unsigned int get_nrrd_type<float>()
{
  return nrrdTypeFloat;
}



}  // end namespace SCITeem
