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

// NrrdData.h - interface to Gordon's Nrrd class
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//
//  Copyright (C) 2001 SCI Institute

#if !defined(SCI_Teem_NrrdData_h)
#define SCI_Teem_NrrdData_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Field.h>
#include <Core/Containers/LockingHandle.h>
#include <nrrd.h>

namespace SCITeem {

using namespace SCIRun;

/////////
// Structure to hold NrrdData
class NrrdData : public Datatype {
public:  
  // GROUP: public data
  //////////
  // 
  Nrrd *nrrd;

  NrrdData(bool owned = true);
  NrrdData(const NrrdData&);
  ~NrrdData();

  virtual NrrdData* clone();

  void set_orig_field(FieldHandle fh) { originating_field_ = fh; }
  FieldHandle get_orig_field() { return originating_field_; }

  //! Is a sci nrrd if we wrap a field up with it, and we have a tuple axis.
  bool is_sci_nrrd() const;
  void copy_sci_data(const NrrdData &);
 
  int get_tuple_axis_size() const;
  bool get_tuple_indecies(vector<string> &elems) const;
  bool get_tuple_index_info(int tmin, int tmax, int &min, int &max) const;
  string concat_tuple_types() const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:
  bool verify_tuple_label(const string &s, vector<string> &elems) const;
  bool in_name_set(const string &s) const;
  bool in_type_set(const string &s) const;

  //! did we wrap some existing memory, or was this allocated
  //! for this object to delete.
  bool                 data_owned_;
  //! a handle to the mesh this data originally belonged with. 
  //! has a rep == 0 if there was no such mesh.
  FieldHandle           originating_field_; 

  // To help with pio
  string                nrrd_fname_;

  static void load_valid_tuple_types();
  static vector<string> valid_tup_types_;
};

typedef LockingHandle<NrrdData> NrrdDataHandle;

// some template helpers...


// nrrd Types that we need to convert to:
//  nrrdTypeChar,          
//  nrrdTypeUChar,         
//  nrrdTypeShort,         
//  nrrdTypeUShort,        
//  nrrdTypeInt,           
//  nrrdTypeUInt,          
//  nrrdTypeLLong,         
//  nrrdTypeULLong,        
//  nrrdTypeFloat,         
//  nrrdTypeDouble,

template <class T>
unsigned int get_nrrd_type();

template <>
unsigned int get_nrrd_type<char>();

template <>
unsigned int get_nrrd_type<unsigned char>();

template <>
unsigned int get_nrrd_type<short>();

template <>
unsigned int get_nrrd_type<unsigned short>();

template <>
unsigned int get_nrrd_type<int>();

template <>
unsigned int get_nrrd_type<unsigned int>();

template <>
unsigned int get_nrrd_type<long long>();

template <>
unsigned int get_nrrd_type<unsigned long long>();

template <>
unsigned int get_nrrd_type<float>();

template <class T>
unsigned int get_nrrd_type() {
  return nrrdTypeDouble;
}


} // end namespace SCITeem

#endif // SCI_Teem_NrrdData_h
