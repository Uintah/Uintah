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

// NrrdData.h - interface to Gordon's Nrrd class
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//


#if !defined(SCI_Teem_NrrdData_h)
#define SCI_Teem_NrrdData_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Field.h>
#include <Core/Containers/LockingHandle.h>
#include <teem/nrrd.h>

#include <Core/Datatypes/share.h>

namespace SCIRun {

/////////
// Structure to hold NrrdData
class SCISHARE NrrdData : public PropertyManager {
public:  
  // GROUP: public data
  //////////
  // 
  Nrrd *nrrd_;

  NrrdData();
  NrrdData(Nrrd *nrrd);
  NrrdData(LockingHandle<Datatype> data_owner);
  NrrdData(const NrrdData&);
  virtual ~NrrdData();

  virtual NrrdData* clone();

  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  // Separate raw files.
  void set_embed_object(bool v) { embed_object_ = v; }
  bool get_embed_object() { return embed_object_; }

  void set_filename( string &f )
  { nrrd_fname_ = f; embed_object_ = false; }
  const string get_filename() const { return nrrd_fname_; }

  bool    write_nrrd_;

protected:
  bool    embed_object_;

  bool in_name_set(const string &s) const;

  //! Either the NrrdData owns the data or it wraps this external object.
  LockingHandle<Datatype> data_owner_;

  // To help with pio
  string                nrrd_fname_;

  static Persistent *maker();
};


typedef LockingHandle<NrrdData> NrrdDataHandle;


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


SCISHARE unsigned int string_to_nrrd_type(const string &str);

template <class T>
SCISHARE unsigned int get_nrrd_type();

template <>
SCISHARE unsigned int get_nrrd_type<char>();

template <>
SCISHARE unsigned int get_nrrd_type<unsigned char>();

template <>
SCISHARE unsigned int get_nrrd_type<short>();

template <>
SCISHARE unsigned int get_nrrd_type<unsigned short>();

template <>
SCISHARE unsigned int get_nrrd_type<int>();

template <>
SCISHARE unsigned int get_nrrd_type<unsigned int>();

template <>
SCISHARE unsigned int get_nrrd_type<long long>();

template <>
SCISHARE unsigned int get_nrrd_type<unsigned long long>();

template <>
SCISHARE unsigned int get_nrrd_type<float>();

template <>
SCISHARE unsigned int get_nrrd_type<Tensor>();

template <class T>
unsigned int get_nrrd_type()
{
  return nrrdTypeDouble;
}

SCISHARE void get_nrrd_compile_type( const unsigned int type,
			    string & typeStr,
			    string & typeName );

} // end namespace SCIRun

#endif // SCI_Teem_NrrdData_h
