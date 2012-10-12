/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <map>
#include <sstream>

#include <libxml/tree.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

ProblemSpecP
ProblemSpec::findBlock() const
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::findBlock()");
  const xmlNode* child = d_node->children;
  if (child != 0) {
    if (child->type == XML_TEXT_NODE) {
      child = child->next;
    }
  }
  if (child == NULL) {
     return 0;
  }
  else {
     return scinew ProblemSpec( child, false );
  }
}

ProblemSpecP 
ProblemSpec::findBlock(const string& name) const 
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::findBlock(string)");
  if (d_node == 0) {
    return 0;
  }
  const xmlNode *child = d_node->children;
  while (child != 0) {
    //    string child_name(to_char_ptr(child->name));
    string child_name((const char *)(child->name));
    if (name == child_name) {
      xmlNode* dbl_child = child->children;
      while (dbl_child != 0) {
        dbl_child = dbl_child->next;
      }
      return scinew ProblemSpec( child, false );
    }
    child = child->next;
  }
  return 0;
}

ProblemSpecP 
ProblemSpec::findBlockWithAttribute(const string& name,
                                    const string& attribute) const 
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::findBlockWithAttribute(string,string)");
  
  for (ProblemSpecP ps = this->findBlock(name); ps != 0;
       ps = ps->findNextBlock(name) ) {

    string attr="";
    ps->getAttribute(attribute,attr);
    if (attr.length() > 0) 
      return ps;
    else
      continue;
  }

  return 0;
}

ProblemSpecP 
ProblemSpec::findBlockWithOutAttribute(const string& name) const

{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::findBlockWithOutAttribute(string)");
  
  for (ProblemSpecP ps = this->findBlock(name); ps != 0;
       ps = ps->findNextBlock(name) ) {

    map<string,string> attributes;
    ps->getAttributes(attributes);
    if (attributes.empty()) 
      return ps;
    else
      continue;
  }

  return 0;
}

ProblemSpecP ProblemSpec::findNextBlock() const
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::findNextBlock()");
  const xmlNode* found_node = d_node->next;
  
  if (found_node != 0) {
    if (found_node->type == XML_TEXT_NODE) {
      found_node = found_node->next;
    }
  }
    
  if (found_node == NULL ) {
     return 0;
  }
  else {
     return scinew ProblemSpec( found_node, false );
  }
}

ProblemSpecP
ProblemSpec::findNextBlock(const string& name) const 
{
  MALLOC_TRACE_TAG_SCOPE("findNextBlock(string)");
  // Iterate through all of the child nodes of the next node
  // until one is found that has this name

  const xmlNode* found_node = d_node->next;

  while(found_node != 0){
    //    string c_name(to_char_ptr(found_node->name));
    string c_name((const char *)(found_node->name));
    if (c_name == name) {
      break;
    }

    found_node = found_node->next;
  }
  if (found_node == NULL) {
     return 0;
  }
  else {
     return scinew ProblemSpec( found_node, false );
  }
}

ProblemSpecP
ProblemSpec::findTextBlock()
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::findTextBlock()");
   for (xmlNode* child = d_node->children; child != 0;
        child = child->next) {
     if (child->type == XML_TEXT_NODE) {
       return scinew ProblemSpec( child, false );
      }
   }
   return NULL;
}

string
ProblemSpec::getNodeName() const
{
  //  return string(to_char_ptr(d_node->name));
  return string((const char *)(d_node->name));
}

short
ProblemSpec::getNodeType() 
{
  return d_node->type;
}

ProblemSpecP
ProblemSpec::importNode(ProblemSpecP src, bool deep) 
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::ImportNode()");
  xmlNode* d = xmlDocCopyNode(src->d_node, d_node->doc, deep ? 1 : 0);
  if (d)
    return scinew ProblemSpec(d, false );
  else
    return 0;
}

void
ProblemSpec::addComment( string comment )
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::addComment()");
  xmlNodePtr commentNode = xmlNewComment(BAD_CAST comment.c_str());
  xmlAddChild(d_node, commentNode);
}

ProblemSpecP
ProblemSpec::makeComment( string comment )
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::makeComment()");
  xmlNodePtr commentNode = xmlNewComment(BAD_CAST comment.c_str());
  return scinew ProblemSpec( commentNode, false );
}


void
ProblemSpec::replaceChild(ProblemSpecP toreplace, 
                          ProblemSpecP replaced) 
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::replaceChild()");
  xmlNode* d = xmlReplaceNode(toreplace->d_node, replaced->d_node);

  if (d)
    xmlFreeNode(d);
}

void
ProblemSpec::removeChild(ProblemSpecP child)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::removeChild()");
  xmlUnlinkNode(child->getNode());
  xmlFreeNode(child->getNode());
}

//______________________________________________________________________
//
void
ProblemSpec::checkForInputError( const string & stringValue, 
                                 ProblemSpec::CheckType type )
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::checkForInputError()");
  //__________________________________
  //  Make sure stringValue only contains valid characters
  switch( type ) {
  case INT_TYPE :
    {
      string validChars(" -0123456789");
      string::size_type  pos = stringValue.find_first_not_of(validChars);
      if (pos != string::npos){
        ostringstream warn;
        warn << "Bad Integer string: Found '"<< stringValue[pos]
             << "' in the string \""<< stringValue<< "\" at position " << pos << ".\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__,true);
      }
    }
    break;
  case FLOAT_TYPE :
    {
      string validChars(" -+.0123456789eE");
      string::size_type  pos = stringValue.find_first_not_of(validChars);
      if (pos != string::npos){
        ostringstream warn;
        warn << "Bad Float string: Found '"<< stringValue[pos]
             << "' inside of \""<< stringValue << "\" at position " << pos << "\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__,true);
      }
      //__________________________________
      // check for two or more "."
      string::size_type p1 = stringValue.find_first_of(".");    
      string::size_type p2 = stringValue.find_last_of(".");     
      if (p1 != p2){
        ostringstream warn;
        warn << "Input file error: I found two (..) "
             << "inside of "<< stringValue << "\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__,true);
      }
    }
    break;
  } // end switch( type )
} 

ProblemSpecP
ProblemSpec::get(const string& name, double &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    checkForInputError( stringValue, FLOAT_TYPE ); 
    istringstream ss(stringValue);
    ss >> value;
    if( !ss ) {
      ps = 0;
      //      cout << "WARNING: ProblemSpec.cc: get(%s, double): stringstream failed..." << name << endl;
    }
  }
          
  return ps;
}

ProblemSpecP
ProblemSpec::get(const string& name, unsigned int &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    checkForInputError( stringValue, INT_TYPE ); 
    istringstream ss(stringValue);
    ss >> value;
    if( !ss ) {
      printf( "WARNING: ProblemSpec.cc: get(%s, uint): stringstream failed...\n", name.c_str() );
      ps = 0;
    }
  }
          
  return ps;

}

ProblemSpecP
ProblemSpec::get(const string& name, int &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    checkForInputError( stringValue, INT_TYPE );
    istringstream ss(stringValue);
    ss >> value;
    if( !ss ) {
      printf( "WARNING: ProblemSpec.cc: get(%s, int): stringstream failed...\n", name.c_str() );
      ps = 0;
    }
  }

  return ps;
}


ProblemSpecP
ProblemSpec::get(const string& name, long &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    checkForInputError( stringValue, INT_TYPE );
    istringstream ss(stringValue);
    ss >> value;
    if( !ss ) {
      printf( "WARNING: ProblemSpec.cc: get(%s, long): stringstream failed...\n", name.c_str() );
      ps = 0;
    }
  }

  return ps;
}

ProblemSpecP
ProblemSpec::get(const string& name, bool &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    // Slurp up any spaces that were put in before or after the cmp string.
    istringstream result_stream(stringValue);
    string nospace_cmp;
    result_stream >> nospace_cmp;

    if( !result_stream ) {
      printf( "WARNING: ProblemSpec.cc: get(%s, bool): stringstream failed...\n", name.c_str() );
    }

    if (nospace_cmp == "false") {
      value = false;
    }
    else if  (nospace_cmp == "true") {
      value = true;
    } else {
      string error = name + " Must be either true or false";
      throw ProblemSetupException(error, __FILE__, __LINE__);
    }
  }
  return ps;

}

ProblemSpecP
ProblemSpec::get(const string& name, string &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  // the other gets will call this one to get the string...
  ProblemSpecP ps = this;
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {  // eliminate spaces
    value = node->getNodeValue();
   
    // elminate spaces from string

    stringstream in_stream(value);
    vector<string> vs;
    copy( istream_iterator<string>(in_stream),
          istream_iterator<string>(),back_inserter(vs) );
    string out_string;
    for (vector<string>::const_iterator it = vs.begin(); it != vs.end();
         ++it) {
      out_string += *it + ' ';
    }

    if (out_string.length() > 0) {
      // if user accidentally leaves out value, this will crash with an ugly exception
      string::iterator begin = out_string.end() - 1;
      string::iterator end = out_string.end();
      out_string.erase(begin,end);
    }
    value = out_string;
  }
  return ps;
}

ProblemSpecP
ProblemSpec::get(const string& name, Point &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
    Vector v;
    ProblemSpecP ps = get(name, v);
    value = Point(v);
    return ps;
}

ProblemSpecP
ProblemSpec::get(const string& name, Vector &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    // Parse out the [num,num,num]
    // Now pull apart the stringValue
    string::size_type i1 = stringValue.find("[");
    string::size_type i2 = stringValue.find_first_of(",");
    string::size_type i3 = stringValue.find_last_of(",");
    string::size_type i4 = stringValue.find("]");
    
    string x_val(stringValue,i1+1,i2-i1-1);
    string y_val(stringValue,i2+1,i3-i2-1);
    string z_val(stringValue,i3+1,i4-i3-1);
    
    checkForInputError( x_val, FLOAT_TYPE ); 
    checkForInputError( y_val, FLOAT_TYPE );
    checkForInputError( z_val, FLOAT_TYPE );
    
    value.x(atof(x_val.c_str()));
    value.y(atof(y_val.c_str()));
    value.z(atof(z_val.c_str()));   
  }
          
  return ps;
}

ProblemSpec::InputType
ProblemSpec::getInputType(const std::string& stringValue) {
  std::string validChars(" +-.0123456789");
  string::size_type  pos = stringValue.find_first_not_of(validChars);
  if (pos != string::npos) {
    // we either have a string or a vector
    if ( stringValue.find_first_of("[") == 0 ) {
      // this is most likely a vector vector
      return ProblemSpec::VECTOR_TYPE;
    } else {
      // we have a string
      return ProblemSpec::STRING_TYPE;
    }
  } else {
    // otherwise we have a number
    return ProblemSpec::NUMBER_TYPE;
  }
  return ProblemSpec::UNKNOWN_TYPE;
}


// value should probably be empty before calling this...
ProblemSpecP
ProblemSpec::get(const string& name, vector<double>& value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  vector<string> string_values;
  if(!this->get(name, string_values)) {
    return 0;
  }
  
  for(vector<string>::const_iterator vit(string_values.begin());
      vit!=string_values.end();vit++) {
    const string v(*vit);
    
    checkForInputError( v, FLOAT_TYPE ); 
    value.push_back( atof(v.c_str()) );
  }
  
  return this;
}

// value should probably be empty before calling this...
ProblemSpecP
ProblemSpec::get(const string& name, vector<int>& value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  vector<string> string_values;
  if(!this->get(name, string_values)) {
    return 0;
  }
  
  for(vector<string>::const_iterator vit(string_values.begin());
      vit!=string_values.end();vit++) {
    const string v(*vit);
    
    checkForInputError( v, FLOAT_TYPE ); 
    value.push_back( atoi(v.c_str()) );
  }
  
  return this;
} 

// value should probably be empty before calling this...
ProblemSpecP
ProblemSpec::get(const string& name, vector<string>& value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    istringstream in(stringValue);
    char c,next;
    string result;
    while (!in.eof()) {
      in >> c;
      if (c == '[' || c == ',' || c == ' ' || c == ']')
        continue;
      next = in.peek();
      result += c;
      if (next == ',' ||  next == ' ' || next == ']') {
        // push next string onto stack
        value.push_back(result);
        result.erase();
      }
    }
  }
  return ps;
} 

ProblemSpecP
ProblemSpec::get(const string& name, IntVector &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    parseIntVector(stringValue, value);
  }

  return ps;
}

ProblemSpecP
ProblemSpec::get(const string& name, vector<IntVector>& value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  ProblemSpecP ps;

  string stringValue;
  ps = get(name, stringValue);
  if (ps == 0) {
    return ps;
  }
  else {
    istringstream in(stringValue);
    char c;
    bool first_bracket = false;
    bool inner_bracket = false;
    string result;
    // what we're going to do is look for the first [ then pass it.
    // then if we find another [, make a string out of it until we see ],
    // then pass that into parseIntVector, and repeat.
    while (!in.eof()) {
      in >> c;
      if (c == ' ' || (c == ',' && !inner_bracket))
        continue;
      if (c == '[') {
        if (!first_bracket) {
          first_bracket = true;
          continue;
        }
        else {
          inner_bracket = true;
        }
      }
      else if (c == ']') {
        if (inner_bracket) {
          // parse the string for an IntVector
          IntVector val;
          result += c;
          // it should be [num,num,num] by now
          parseIntVector(result, val);
          value.push_back(val);
          result.erase();
          inner_bracket = false;
          continue;
        }
        else
          break; // end parsing on outer ]
      }
      // add the char to the string
      result += c;
    }  // end while (!in.eof())
  }
  
  return ps;
}

void ProblemSpec::parseIntVector(const string& string_value, IntVector& value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  // Parse out the [num,num,num]
  // Now pull apart the string_value
  string::size_type i1 = string_value.find("[");
  string::size_type i2 = string_value.find_first_of(",");
  string::size_type i3 = string_value.find_last_of(",");
  string::size_type i4 = string_value.find("]");
  
  string x_val(string_value,i1+1,i2-i1-1);
  string y_val(string_value,i2+1,i3-i2-1);
  string z_val(string_value,i3+1,i4-i3-1);

  checkForInputError( x_val, INT_TYPE );
  checkForInputError( y_val, INT_TYPE );
  checkForInputError( z_val, INT_TYPE );
          
  value.x(atoi(x_val.c_str()));
  value.y(atoi(y_val.c_str()));
  value.z(atoi(z_val.c_str())); 

}

bool
ProblemSpec::get(int &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  string stringValue;
  if (!get(stringValue))
    return false;
  value = atoi(stringValue.c_str());
  return true;
}

bool
ProblemSpec::get(long &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  string stringValue;
  if (!get(stringValue))
    return false;
  value = atoi(stringValue.c_str());
  return true;
}

bool
ProblemSpec::get(double &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  string stringValue;
  if (!get(stringValue))
    return false;
  value = atof(stringValue.c_str());
  return true;
}

bool
ProblemSpec::get(string &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  string tmp = getNodeValue();
  if (tmp == "")
    return false;
  istringstream tmp_str(tmp);
  string w;
  while(tmp_str>>w) value += w;
  return true;
}

bool
ProblemSpec::get(Vector &value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::get()");
  string stringValue;
  if (!get(stringValue))
    return false;
  // Now pull apart the stringValue
  string::size_type i1 = stringValue.find("[");
  string::size_type i2 = stringValue.find_first_of(",");
  string::size_type i3 = stringValue.find_last_of(",");
  string::size_type i4 = stringValue.find("]");
  
  string x_val(stringValue,i1+1,i2-i1-1);
  string y_val(stringValue,i2+1,i3-i2-1);
  string z_val(stringValue,i3+1,i4-i3-1);
  
  checkForInputError( x_val, FLOAT_TYPE );
  checkForInputError( y_val, FLOAT_TYPE );
  checkForInputError( z_val, FLOAT_TYPE );
  
  value.x(atof(x_val.c_str()));
  value.y(atof(y_val.c_str()));
  value.z(atof(z_val.c_str()));      
  return true;
}


ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            double& value, double defaultVal) 
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create xmlNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }

  return ps;
}

ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            int& value, int defaultVal)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create xmlNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value=defaultVal;
  }

  return ps;
}
ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            bool& value, bool defaultVal)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create xmlNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value=defaultVal;
  }

  return ps;
}
ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            string& value, 
                            const string& defaultVal)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create xmlNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }
  return ps;
}
ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            IntVector& value, 
                            const IntVector& defaultVal)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create xmlNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }

  return ps;
}

ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            Vector& value, 
                            const Vector& defaultVal)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create xmlNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }

  return ps;
}

ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            Point& value, 
                            const Point& defaultVal)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create xmlNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }

  return ps;
}

ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            vector<double>& value, 
                            const vector<double>& defaultVal)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  value.clear();
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create xmlNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;

    value.clear();
    int size = static_cast<int>(defaultVal.size());
    for (int i = 0; i < size; i++)
      value.push_back(defaultVal[i]);
  }

  return ps;
}

ProblemSpecP
ProblemSpec::getWithDefault(const string& name, 
                            vector<int>& value, 
                            const vector<int>& defaultVal)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getWithDefault()");
  value.clear();
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    // add xmlNode to the tree
    appendElement(name.c_str(), defaultVal);
    // set default values
    ps = this;
    value.clear();
    int size = static_cast<int>(defaultVal.size());
    for (int i = 0; i < size; i++)
      value.push_back(defaultVal[i]);
  }

  return ps;
}

ProblemSpecP
ProblemSpec::appendElement(const char* name, const string& value)
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::appendElement()");
  xmlNode* newnode = xmlNewChild(d_node, 0, BAD_CAST name, BAD_CAST value.c_str());
  return scinew ProblemSpec( newnode, false );
}

//basically to make sure correct overloaded function is called
ProblemSpecP
ProblemSpec::appendElement(const char* name, const char* value)
{
  return appendElement(name, string(value));
}

ProblemSpecP
ProblemSpec::appendElement(const char* name, int value)
{
  ostringstream val;
  val << value;
  return appendElement(name, val.str());
}

ProblemSpecP
ProblemSpec::appendElement(const char* name, long value)
{
  ostringstream val;
  val << value;
  return appendElement(name, val.str());
}

ProblemSpecP
ProblemSpec::appendElement(const char* name, const IntVector& value)
{
  ostringstream val;
  val << '[' << value.x() << ", " << value.y() << ", " << value.z() << ']';
  return appendElement(name, val.str());
}

ProblemSpecP
ProblemSpec::appendElement(const char* name, const Point& value)
{

  ostringstream val;
  val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " 
      << setprecision(17) << value.z() << ']';
  return appendElement(name, val.str());

}

ProblemSpecP
ProblemSpec::appendElement(const char* name, const Vector& value)
{
   ostringstream val;
   val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", "
       << setprecision(17) << value.z() << ']';
   return appendElement(name, val.str());

}

ProblemSpecP
ProblemSpec::appendElement(const char* name, double value )
{
   ostringstream val;
   val << setprecision(17) << value;
   return appendElement(name, val.str());

}

ProblemSpecP
ProblemSpec::appendElement( const char* name, const vector<double>& value)
{
   ostringstream val;
   val << '[';
   for (unsigned int i = 0; i < value.size(); i++) {
     val << setprecision(17) << value[i];
     if (i !=  value.size()-1)
       val << ',';
     
   }
   val << ']';
   return appendElement(name, val.str());

}

ProblemSpecP
ProblemSpec::appendElement(const char* name, const vector<int>& value)
{
   ostringstream val;
   val << '[';
   for (unsigned int i = 0; i < value.size(); i++) {
     val << setprecision(17) << value[i];
     if (i !=  value.size()-1)
       val << ',';
     
   }
   val << ']';
   return appendElement(name, val.str());
}

ProblemSpecP
ProblemSpec::appendElement(const char* name, const vector<string >& value)
{
   ostringstream val;
   val << '[';
   for (unsigned int i = 0; i < value.size(); i++) {
     val <<  value[i];
     if (i !=  value.size()-1)
       val << ',';
     
   }
   val << ']';
   return appendElement(name, val.str());
}



ProblemSpecP
ProblemSpec::appendElement( const char* name, bool value )
{
  if (value)
    return appendElement(name, string("true"));
  else
    return appendElement(name, string("false"));
}

void
ProblemSpec::require(const string& name, double& value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::require(const string& name, int& value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::require(const string& name, unsigned int& value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
      throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::require(const string& name, long& value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::require(const string& name, bool& value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
      throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::require(const string& name, string& value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::require(const string& name, Vector  &value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
   throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::require(const string& name, vector<double>& value)
{

  // Check if the prob_spec is NULL

  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);

}

void
ProblemSpec::require(const string& name, vector<string>& value)
{
  
  // Check if the prob_spec is NULL
  
  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);
  
}


void
ProblemSpec::require(const string& name, vector<int>& value)
{

  // Check if the prob_spec is NULL

  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);

} 

void
ProblemSpec::require(const string& name, vector<IntVector>& value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);
} 

void
ProblemSpec::require(const string& name, IntVector  &value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::require(const string& name, Point  &value)
{
  // Check if the prob_spec is NULL
  if (! this->get(name,value))
    throw ParameterNotFound(name, __FILE__, __LINE__);
}

void
ProblemSpec::getAttributes(map<string,string>& attributes) const
{
  attributes.clear();

  xmlAttr* attr = d_node->properties;

  for (; attr != 0; attr = attr->next) {
    if (attr->type == XML_ATTRIBUTE_NODE) {
      attributes[(const char *)(attr->name)] = (const char *)(attr->children->content);
    }
  }
}

bool
ProblemSpec::getAttribute(const string& attribute, string& result) const
{

  map<string, string> attributes;
  getAttributes(attributes);

  map<string,string>::iterator iter = attributes.find(attribute);

  if (iter != attributes.end()) {
    result = iter->second;
    return true;
  }
  else {
    return false;
  }
}

bool
ProblemSpec::getAttribute(const string& name, double &value) const
{
  string stringValue;
  if(!getAttribute(name, stringValue)) {
    return false;
  }
  checkForInputError( stringValue, FLOAT_TYPE ); 
  istringstream ss(stringValue);
  ss >> value;
  if( !ss ) {
    printf( "WARNING: ProblemSpec.cc: getAttribute(%s, double): stringstream failed...\n", name.c_str() );
  }
          
  return true;
}


bool
ProblemSpec::getAttribute(const string& attribute, std::vector<std::string>& result) const
{
  
  map<string, string> attributes;
  getAttributes(attributes);
  
  map<string,string>::iterator iter = attributes.find(attribute);
  
  if (iter != attributes.end()) {
    std::string attributeName = iter->second;
    std::stringstream ss(attributeName);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    result.assign(begin,end);
    //std::vector<std::string> vstrings(begin, end);
    //result = iter->second;
    return true;
  }
  else {
    return false;
  }
}


void
ProblemSpec::setAttribute(const string& name, 
                          const string& value)
{
  xmlNewProp(d_node, BAD_CAST name.c_str(), BAD_CAST value.c_str());
}


ProblemSpecP
ProblemSpec::getFirstChild() 
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getFirstChild()");
  xmlNode* d = d_node->children;
  if (d) {
    return scinew ProblemSpec( d, false );
  }
  else {
    return 0;
  }
}

ProblemSpecP
ProblemSpec::getNextSibling() 
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getNextSibling()");
  xmlNode* d = d_node->next;
  if( d ) {
    return scinew ProblemSpec( d, false );
  }
  else {
    return 0;
  }
}

string
ProblemSpec::getNodeValue() 
{
  string ret;
  for (xmlNode *child = d_node->children; child != 0;
       child = child->next) {
    if (child->type == XML_TEXT_NODE) {
      ret = (const char *)(child->content);
      break;
    }
  }
  return ret;
}

// append element with associated string
ProblemSpecP
ProblemSpec::appendChild( const char *str )
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::appendChild()");
  xmlNode* elt = xmlNewChild(d_node, 0, BAD_CAST str, 0);
  
  return scinew ProblemSpec( elt, false );
}

void
ProblemSpec::appendChild( ProblemSpecP pspec )
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::appendChild()");
  xmlAddChild(d_node, pspec->d_node);
}

// filename is a default param (NULL)
//   call with no parameters or NULL to output
//   to stdout
void
ProblemSpec::output(const char* filename) const 
{
  if (filename) {
    xmlKeepBlanksDefault(0);
    xmlSaveFormatFileEnc(filename, d_node->doc, "UTF-8", 1);
  }
}

void
ProblemSpec::releaseDocument() 
{
  xmlFreeDoc(d_node->doc);
}

ProblemSpecP
ProblemSpec::getRootNode()
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::getRootNode()");
  xmlNode* root_node = xmlDocGetRootElement(d_node->doc);
  return scinew ProblemSpec( root_node, false ); // don't mark as toplevel as this is just a copy
}

const Uintah::TypeDescription*
ProblemSpec::getTypeDescription()
{
  //cerr << "ProblemSpec::getTypeDescription() not done\n";
  return 0;
}

// static
ProblemSpecP
ProblemSpec::createDocument(const string& name) 
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpec::createDocument()");
  xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
  xmlNodePtr node = xmlNewDocRawNode(doc, 0, BAD_CAST name.c_str(), 0);

  xmlDocSetRootElement(doc, node);

  return scinew ProblemSpec(node, true );
}

string
ProblemSpec::getFile() const
{
  return (const char *)( d_node->doc->URL );
}
