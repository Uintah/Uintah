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
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cerr;

namespace SCITeem {

using namespace SCIRun;

static Persistent* make_NrrdData() {
  return scinew NrrdData;
}

PersistentTypeID NrrdData::type_id("NrrdData", "PropertyManager", make_NrrdData);

vector<string> NrrdData::valid_tup_types_;


NrrdData::NrrdData(bool owned) : 
  nrrd(nrrdNew()),
  data_owned_(owned)
{
  if (valid_tup_types_.size() == 0) {
    load_valid_tuple_types();
  }
}

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

void 
NrrdData::load_valid_tuple_types() 
{
  valid_tup_types_.push_back("Scalar");
  valid_tup_types_.push_back("Vector");
  valid_tup_types_.push_back("Tensor");
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


// This would be much easier to check with a regular expression lib
// A valid label has the following format:
// type = one of the valid types (Scalar, Vector, Tensor)
// elem = [A-Za-z0-9\-]+:type
// (elem,?)+

bool 
NrrdData::in_name_set(const string &s) const
{
  const string 
    word("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_");
  
  //cout << "checking in_name " << s << endl;
  // test against valid char set.

  for(string::size_type i = 0; i < s.size(); i++) {
    bool in_set = false;
    for (unsigned int c = 0; c < word.size(); c++) {
      if (s[i] == word[c]) {
	in_set = true;
	break;
      }
    }
    if (! in_set) {
      //cout << "in_name_set failing" << endl;
      return false;
    }
  }

  return true;
}

bool 
NrrdData::in_type_set(const string &s) const
{
  // test against valid char set.
  vector<string>::iterator iter = valid_tup_types_.begin();
  while (iter != valid_tup_types_.end()) {
    //cout << "comp " << s << " " << *iter << endl;
    if (s == *iter) {
      return true;
    }
    ++iter;
  }
  //cout << "in_type_set failing" << endl;
  return false;
}


bool
NrrdData::verify_tuple_label(const string &s, vector<string> &elems) const
{

  // first char must be part of name set
  string::size_type nm_idx = 0;
  string::size_type type_idx = s.size();

  if (! s.size()) return false;

  //cout << "label is: " << s << endl;
  for(string::size_type i = 0; i < s.size(); i++) {
    //cout << s[i] << endl;
    if (s[i] == ':') {
      // substring up until here must be a name
      string sub = s.substr(nm_idx, i - nm_idx);
      if (! in_name_set(sub)) return false;
      // set nm_idx to something invalid for now.
      type_idx = i+1;

    } else if (s[i] == ',' || (i == s.size() - 1)) {
      int off = 0;
      if (i == s.size() - 1) {
	off = 1;
      }
      // substring up until here must be an elem
      //cout << "sub from : " << type_idx << " to: " << i << endl;
      string sub = s.substr(type_idx, i - type_idx + off);
      if (! in_type_set(sub)) return false;
      // the valid elem is from nm_idx to i-1
      string elem = s.substr(nm_idx, i - nm_idx + off);
      elems.push_back(elem);

      // start looking for next valid elem
      nm_idx = i+1;
      // set type_idx to something invalid for now.
      type_idx = s.size();
    }
  }
  return true;
}


// return a comma separated list of just the type names along the tuple axis.
string
NrrdData::concat_tuple_types() const
{
  string rval;
  const string s(nrrd->axis[0].label);

  // first char must be part of name set
  string::size_type nm_idx = 0;
  string::size_type type_idx = s.size();

  if (! s.size()) return false;

  //cout << "label is: " << s << endl;
  for(string::size_type i = 0; i < s.size(); i++) {
    //cout << s[i] << endl;
    if (s[i] == ':') {
      // substring up until here must be a name
      string sub = s.substr(nm_idx, i - nm_idx);
      if (! in_name_set(sub)) return false;
      // set nm_idx to something invalid for now.
      type_idx = i+1;

    } else if (s[i] == ',' || (i == s.size() - 1)) {
      int off = 0;
      if (i == s.size() - 1) {
	off = 1;
      }
      // substring up until here must be an elem
      //cout << "sub from : " << type_idx << " to: " << i << endl;
      string sub = s.substr(type_idx, i - type_idx + off);
      if (rval.size() == 0) {
	rval = sub;
      } else {
	rval += string(",") + sub;
      }
      // start looking for next valid elem
      nm_idx = i+1;
      // set type_idx to something invalid for now.
      type_idx = s.size();
    }
  }
  return rval;
}

bool
NrrdData::get_tuple_indecies(vector<string> &elems) const
{
  if (!nrrd) return false;
  string tup(nrrd->axis[0].label);
  return verify_tuple_label(tup, elems);
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

#define NRRDDATA_VERSION 3

//////////
// PIO for NrrdData objects
void NrrdData::io(Piostream& stream) {
  int version =  stream.begin_class("NrrdData", NRRDDATA_VERSION);
  // Do the base class first...
  if (version > 2) {
    PropertyManager::io(stream);
  }

  if (stream.reading()) {
    Pio(stream, nrrd_fname_);
    if (nrrdLoad(nrrd = nrrdNew(), strdup(nrrd_fname_.c_str()), nrrdIONew())) {
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
    NrrdIO *no = 0;
    TextPiostream *text = dynamic_cast<TextPiostream*>(&stream);
    if (text) {
      no = nrrdIONew();
      no->encoding = nrrdEncodingAscii;
    } 
    if (nrrdSave(strdup(nrrd_fname_.c_str()), nrrd, no)) {
      char *err = biffGet(NRRD);      
      cerr << "Error writing nrrd " << nrrd_fname_ << ": "<< err << endl;
      free(err);
      biffDone(NRRD);
      return;
    }
    if (text) { nrrdIONix(no); }
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
