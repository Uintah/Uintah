/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <teem/nrrd.h>

namespace SCIRun {

/////////
// Structure to hold NrrdData
class NrrdData : public PropertyManager {
public:  
  // GROUP: public data
  //////////
  // 
  Nrrd *nrrd;

  NrrdData(bool owned = true);
  NrrdData(const NrrdData&);
  ~NrrdData();

  virtual NrrdData* clone();

  //void set_orig_field(FieldHandle fh) { originating_field_ = fh; }
  //FieldHandle get_orig_field() { return originating_field_; }

  //! Is a sci nrrd if we wrap a field up with it, and we have a tuple axis.
  //bool is_sci_nrrd() const;
  //void copy_sci_data(const NrrdData &);
 
  //int get_tuple_axis_size() const;
  //bool get_tuple_indecies(vector<string> &elems) const;
  //bool get_tuple_index_info(int tmin, int tmax, int &min, int &max) const;
  //string concat_tuple_types() const;
  //bool verify_tuple_label(const string &s, vector<string> &elems) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  // Separate raw files.
  void set_embed_object(bool v) { embed_object_ = v; }
  bool get_embed_object() { return embed_object_; }
  void set_filename( string &f )
  { nrrd_fname_ = f; embed_object_ = false; }
  const string get_filename() const { return nrrd_fname_; }

protected:
  bool    embed_object_;

private:
  bool in_name_set(const string &s) const;
  //bool in_type_set(const string &s) const;

  //! did we wrap some existing memory, or was this allocated
  //! for this object to delete.
  bool                 data_owned_;
  //! a handle to the mesh this data originally belonged with. 
  //! has a rep == 0 if there was no such mesh.
  //FieldHandle           originating_field_; 



  // To help with pio
  public:
  string                nrrd_fname_;

  //static void load_valid_tuple_types();
  //static vector<string> valid_tup_types_;
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

template <>
unsigned int get_nrrd_type<Tensor>();

template <class T>
unsigned int get_nrrd_type() {
  return nrrdTypeDouble;
}

} // end namespace SCIRun

#endif // SCI_Teem_NrrdData_h
