/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef Uintah_Package_Core_Datatypes_VariableCache_h
#define Uintah_Package_Core_Datatypes_VariableCache_h 1

/*
 *  VariableCache.h:  Cache variables extracted from a Grid
 *
 *  Stores the results of queried variables, to be accessed later.
 *  
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */

#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/Matrix3.h>
#include <vector>
#include <string>
#include <map>

namespace Uintah {

  using namespace SCIRun;

class VariableCache: public Datatype {
protected:
  std::map< std::string, std::string > data_cache;
public:

  // empties cache
  void clear() { data_cache.clear(); }
  
  // Tells you if the variable has been cached.
  bool is_cached(const std::string &name);
  
  // Returns true if name was found and sets data with the cached values
  // Returns false otherwise (not setting data with anything).
  bool get_cached(const std::string &name, std::string& data);

  // The vector of values are put into std::string form and assigned to data.
  // data is then cached based on key.
  void cache_value(const std::string &key, std::vector< int > &values, std::string &data);
  void cache_value(const std::string &key, std::vector< std::string > &values, std::string &data);
  void cache_value(const std::string &key, std::vector< double > &values, std::string &data);
  void cache_value(const std::string &key, std::vector< float > &values, std::string &data);

  // Scalar values based on Vector are cached.
  // 
  // These types are               cached under
  // ----------------             --------------
  //     values[i].length           key+" length"
  //     values[i].length2          key+" leghth2"
  //     values[i].x                key+" x"
  //     values[i].y                key+" y"
  //     values[i].z                key+" z"
  //
  // To access, for example, the legth of variable stored under key="myvar"
  // call is_cached(std::string("myvar"+" length"), data);
  // 
  void cache_value(const std::string &key, std::vector< Vector > &values);

  // Scalar values based on Matrix3 are cached.
  // 
  // These types are               cached under
  // ----------------             --------------
  //     values[i].Determinant      key+" Determinant"
  //     values[i].Trace            key+" Trace"
  //     values[i].Norm             key+" Norm"
  //
  void cache_value(const std::string &key, std::vector< Matrix3 > &values);

  // Converts the vector of data to a std::string seperated by spaces
  static std::string vector_to_string(std::vector< int > &data);
  static std::string vector_to_string(std::vector< std::string > &data);
  static std::string vector_to_string(std::vector< double > &data);
  static std::string vector_to_string(std::vector< float > &data);
  // These take an extra argument witch tells what kind of scalar to extract
  // type can be: "length", "length2", "x", "y", or "z"
  static std::string vector_to_string(std::vector< Vector > &data, const std::string &type);
  // type can be: "Determinant", "Trace", "Norm"
  static std::string vector_to_string(std::vector< Matrix3 > &data, const std::string &type);

  //////////////////////////////////////////////////////////
  // inherited functions
  virtual void io(Piostream&);
};

} // end namespace Uintah

#endif // Uintah_Package_Core_Datatypes_VariableCache_h
