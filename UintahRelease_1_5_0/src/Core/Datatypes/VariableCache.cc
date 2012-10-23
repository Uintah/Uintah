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

// Header file for .cc file
#include <Core/Datatypes/VariableCache.h>

// Local includes
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Math/Matrix3.h>

// Basic includes
#include <sstream>
#include <vector>
#include <string>


using namespace SCIRun;
using namespace Uintah;


// Tells you if the variable has been cached.
bool VariableCache::is_cached(const string &name) {
  map< string, string >::iterator iter;
  iter = data_cache.find(name);
  if (iter == data_cache.end()) {
    return false;
  }
  else {
    return true;
  }
}

// Tells you if the variable has been cached.
// Returns true if name was found and sets data with the cached values
// Returns false otherwise (not setting data with anything).
bool VariableCache::get_cached(const string &name, string& data) {
  map< string, string >::iterator iter;
  iter = data_cache.find(name);
  if (iter == data_cache.end()) {
    return false;
  }
  else {
    data = iter->second;
    return true;
  }
}

// The vector of values are put into string form and assigned to data.
// data is then cached based on key.
void VariableCache::cache_value(const string &key, vector< int > &values,
				string &data) {
  data = vector_to_string(values);
  data_cache[key] = data;
}

void VariableCache::cache_value(const string &key, vector< string > &values,
				string &data) {
  data = vector_to_string(values);
  data_cache[key] = data;
}

void VariableCache::cache_value(const string &key, vector< double > &values,
				string &data) {
  data = vector_to_string(values);
  data_cache[key] = data;
}  

void VariableCache::cache_value(const string &key, vector< float > &values,
				string &data) {
  data = vector_to_string(values);
  data_cache[key] = data;
}  

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
void VariableCache::cache_value(const string &key, vector< Vector > &values) {
  string data = vector_to_string(values,"length");
  data_cache[key+" length"] = data;
  data = vector_to_string(values,"length2");
  data_cache[key+" length2"] = data;
  data = vector_to_string(values,"x");
  data_cache[key+" x"] = data;
  data = vector_to_string(values,"y");
  data_cache[key+" y"] = data;
  data = vector_to_string(values,"z");
  data_cache[key+" z"] = data;
}

// Scalar values based on Matrix3 are cached.
// 
// These types are               cached under
// ----------------             --------------
//     values[i].Determinant      key+" Determinant"
//     values[i].Trace            key+" Trace"
//     values[i].Norm             key+" Norm"
//
void VariableCache::cache_value(const string &key, vector< Matrix3 > &values) {
  string data = vector_to_string(values,"Determinant");
  data_cache[key+" Determinant"] = data;
  data = vector_to_string(values,"Trace");
  data_cache[key+" Trace"] = data;
  data = vector_to_string(values,"Norm");
  data_cache[key+" Norm"] = data;
}

// Converts the vector of data to a string seperated by spaces
string VariableCache::vector_to_string(vector< int > &data) {
  std::ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string VariableCache::vector_to_string(vector< string > &data) {
  string result;
  for(int i = 0; i < (int)data.size(); i++) {
      result+= (data[i] + " ");
    }
  return result;
}

string VariableCache::vector_to_string(vector< double > &data) {
  std::ostringstream ostr;
  ostr.precision(15);
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string VariableCache::vector_to_string(vector< float > &data) {
  std::ostringstream ostr;
  ostr.precision(15);
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

// These take an extra argument witch tells what kind of scalar to extract
// type can be: "length", "length2", "x", "y", or "z"
string VariableCache::vector_to_string(vector< Vector > &data,
				       const string &type) {
  std::ostringstream ostr;
  ostr.precision(15);
  if (type == "length") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].length() << " ";
    }
  } else if (type == "length2") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].length2() << " ";
    }
  } else if (type == "x") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].x() << " ";
    }
  } else if (type == "y") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].y() << " ";
    }
  } else if (type == "z") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].z() << " ";
    }
  }

  return ostr.str();
}

// type can be: "Determinant", "Trace", "Norm"
string VariableCache::vector_to_string(vector< Matrix3 > &data,
				       const string &type) {
  std::ostringstream ostr;
  ostr.precision(15);
  if (type == "Determinant") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Determinant() << " ";
    }
  } else if (type == "Trace") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Trace() << " ";
    }
  } else if (type == "Norm") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Norm() << " ";
    } 
 }

  return ostr.str();
}

void VariableCache::io(Piostream &) {
  // do nothing
}
