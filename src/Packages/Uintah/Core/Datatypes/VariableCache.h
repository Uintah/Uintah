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
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Persistent/Persistent.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using std::vector;
  using std::string;
  using std::map;
  using namespace SCIRun;

class SCICORESHARE VariableCache: public Datatype {
protected:
  map< string, string > data_cache;
public:

  // empties cache
  void clear() { data_cache.clear(); }
  
  // Tells you if the variable has been cached.
  bool is_cached(const string &name);
  
  // Returns true if name was found and sets data with the cached values
  // Returns false otherwise (not setting data with anything).
  bool get_cached(const string &name, string& data);

  // The vector of values are put into string form and assigned to data.
  // data is then cached based on key.
  void cache_value(const string &key, vector< int > &values, string &data);
  void cache_value(const string &key, vector< string > &values, string &data);
  void cache_value(const string &key, vector< double > &values, string &data);

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
  // call is_cached(string("myvar"+" length"), data);
  // 
  void cache_value(const string &key, vector< Vector > &values);

  // Scalar values based on Matrix3 are cached.
  // 
  // These types are               cached under
  // ----------------             --------------
  //     values[i].Determinant      key+" Determinant"
  //     values[i].Trace            key+" Trace"
  //     values[i].Norm             key+" Norm"
  //
  void cache_value(const string &key, vector< Matrix3 > &values);

  // Converts the vector of data to a string seperated by spaces
  static string vector_to_string(vector< int > &data);
  static string vector_to_string(vector< string > &data);
  static string vector_to_string(vector< double > &data);
  // These take an extra argument witch tells what kind of scalar to extract
  // type can be: "length", "length2", "x", "y", or "z"
  static string vector_to_string(vector< Vector > &data, const string &type);
  // type can be: "Determinant", "Trace", "Norm"
  static string vector_to_string(vector< Matrix3 > &data, const string &type);

  //////////////////////////////////////////////////////////
  // inherited functions
  virtual void io(Piostream&);
};

} // end namespace Uintah

#endif // Uintah_Package_Core_Datatypes_VariableCache_h
