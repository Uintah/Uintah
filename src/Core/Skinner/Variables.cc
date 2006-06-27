//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//  
//    File   : Variables.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:01:09 2006
#include <Core/Skinner/Skinner.h>
#include <Core/Skinner/Variables.h>

#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Environment.h>
#include <Core/Util/Assert.h>

#include <libxml/xmlreader.h>
#include <libxml/catalog.h>
#include <iostream>


namespace SCIRun {
  namespace Skinner {
    Variables::Variables(const string &id) 
      : variables_(),
        parent_(0),
        children_()
    {
      insert("id", id);
    }

    Variables::Variables(const string &id, Variables *parent) 
      : variables_(),
        parent_(parent),
        children_()
    {
      ASSERT(parent);
      insert("id", id);
    }

    Variables::~Variables() 
    {
      children_t::iterator citer = children_.begin();
      while (citer != children_.end()) {
        Variables *child = *(citer++);
        // Must delete child after incrementing pointer
        // because code below will be invoked by child
        // and remove iterator
        delete child;
      }

      if (parent_) {
        parent_->children_.erase(this);
      }
    }

    Variables *
    Variables::spawn(const string &id)
    {
      return new Variables(get_id()+":"+id, this);
    }

    Variables *
    Variables::parent() {
      return parent_;
    }

    void
    Variables::insert(const string &name, 
                      const string &value,
                      const string &type_str,
                      bool propagate)
    {

      var_type_e type_enum = STRING_E;
      if (string_tolower(type_str) == "int") {
        type_enum = INT_E;
      } else if (string_tolower(type_str) == "float") {
        type_enum = DOUBLE_E;
      } else if (string_tolower(type_str) == "double") {
        type_enum = DOUBLE_E;
      } else if (string_tolower(type_str) == "bool") {
        type_enum = BOOL_E;
      }

      name_value_map_t::iterator iter = variables_.find(name);
      if (iter == variables_.end()) {
        variables_.insert(make_pair(name, value_t(value,type_enum,propagate)));
      } else {
        iter->second.value = value;
        iter->second.var_type = type_enum;
        iter->second.propagate = propagate;
      }
    }


    bool
    Variables::maybe_get_int(const string &name, int &val) {
      string str;
      return (maybe_get_string(name, str) && string_to_int(str, val));
    }

    bool
    Variables::maybe_get_double(const string &name, double &val) {
      string str;
      return (maybe_get_string(name, str) && string_to_double(str, val));
    }

    bool
    Variables::maybe_get_string(const string &name, 
                                string &val) const 
    {
      const Variables *this_ptr = this;
      while (this_ptr) {
        name_value_map_t::const_iterator iter =this_ptr->variables_.find(name);
        if (iter != this_ptr->variables_.end()) {
          if (!iter->second.propagate && this_ptr != this) {
            return false;
          }
          val = iter->second.value;
          if (val[0] == '$') {
            string varname = val.substr(1, val.length()-1);
            maybe_get_string(varname, val);
          }
          return true;
        }
        this_ptr = this_ptr->parent_;
      } 

      return false;
    }

    bool
    Variables::maybe_get_color(const string &name, Color &val) {
      string str;
      if (maybe_get_string(name, str)) {
        if (str[0] != '#') {
          return false;
        }

        int r,g,b,a;
        str = string_toupper(str).substr(1, str.length()-1);
        sscanf(str.c_str(), "%02X%02X%02X%02X", &r, &g, &b, &a);

        const double scale = 1.0/255.0;            
        val.r = r*scale;
        val.g = g*scale;
        val.b = b*scale;
        val.a = a*scale;

        return true;
      }

      return false;      
    }


    bool
    Variables::maybe_get_bool(const string &name, bool &val) {
      string str;
      if (maybe_get_string(name, str)) {
        str = string_toupper(str);
        if (str=="0" || str=="F" || str=="OFF" || str=="FALSE" || str=="NO") {
          val = false;
        } else {
          // If the string exists, and is not one of the strings above,
          // assume it is true...similar to C assumes anything not 0
          // is true
          val = true;
        }
        return true;
      }

      return false;
    }



    int 
    Variables::get_int(const string &name) {
      int val;
      if (!maybe_get_int(name, val)) {
        throw "Variables::get_int cannot get " + name;
      }
      return val;
    }

    double 
    Variables::get_double(const string &name) {
      double val;
      if (!maybe_get_double(name, val)) {
        throw "Variables::get_double cannot get " + name;
      }
      return val;
    }
  
    string
    Variables::get_string(const string &name) {
      string val;
      if (!maybe_get_string(name, val)) {
        throw "Variables::get_string cannot get " + name;
      }
      return val;
    }    

    Color
    Variables::get_color(const string &name) {
      Color val;
      if (!maybe_get_color(name, val)) {
        throw "Variables::get_color cannot get " + name;
      }
      return val;
    }

    string
    Variables::get_id() const {
      string id = "";
      maybe_get_string("id", id);
      return id;
    }

    bool
    Variables::get_bool(const string &name) {
      bool val;
      if (maybe_get_bool(name, val)) {
        return val;
      }
      return false;
    }

    Variables::value_t::value_t(string val, var_type_e vtype, bool prop) :
      value(val),
      var_type(vtype),
      propagate(prop) 
      {
      }
    
  }
}
