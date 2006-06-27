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
//    File   : Variables.h
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:03:25 2006

#ifndef Skinner_Variables_H
#define Skinner_Variables_H

#include <Core/Skinner/Color.h>

#include <string>
#include <map>
#include <set>
#include <vector>

using std::string;
using std::map;
using std::vector;
using std::set;


namespace SCIRun {
  namespace Skinner {
    class Variables {
      Variables         (const string &id, Variables *parent);
    public:
      Variables         (const string &id);
      virtual           ~Variables();

      Variables *       spawn(const string &id);
      Variables *       parent();
      void              insert(const string &name,
                               const string &value, 
                               const string &type_str = "string",
                               bool propagate = false);

      string            get_id() const;
      int               get_int(const string &);
      double            get_double(const string &);
      string            get_string(const string &);
      bool              get_bool(const string &);
      Color             get_color(const string &);

      bool              maybe_get_int(const string &, int &);
      bool              maybe_get_double(const string &, double &);
      bool              maybe_get_string(const string &, string &) const;
      bool              maybe_get_color(const string &, Color &);
      bool              maybe_get_bool(const string &, bool &);

    private:
      enum var_type_e {
        STRING_E,
        INT_E,
        DOUBLE_E,
        BOOL_E
      };
      
      struct value_t {
        value_t(string, var_type_e, bool);

        string            value;
        var_type_e        var_type;
        bool              propagate;
      };

      typedef map<string, value_t> name_value_map_t;
      typedef name_value_map_t::value_type entry_t;
      typedef set<Variables *> children_t;

      name_value_map_t  variables_;
      Variables *       parent_;
      children_t        children_;
      

    };
  } // end namespace Skinner
} // end namespace SCIRun


#endif // #define Skinner_Variables_H
