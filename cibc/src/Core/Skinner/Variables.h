//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
#include <Core/Util/Assert.h>

#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <functional>
using std::string;
using std::map;
using std::vector;
using std::pair;

#include <Core/Skinner/share.h>

namespace SCIRun {
  namespace Skinner {
    class Variables;
    template <class T>
    class Var {
    private:
      friend class Variables;
      Variables *       scope_;
      int *             scope_index_;
    public:
      Var(Variables *vars, const string &name);
      Var(Variables *vars, const string &name, const T &);
      Var() : scope_(0), scope_index_(0) {}
      Var(const Var<T> &copy) : 
        scope_(copy.scope_), 
        scope_index_(copy.scope_index_)
      {
      }
      bool exists() { return scope_index_ && (*scope_index_ >= 0); }
      operator T();
      Var<T> & operator= (const T& rhs);
      Var<T> & operator= (const Var<T>& rhs);
      Var<T> & operator|=(const T& rhs);
      Var<T> & operator|=(const Var<T>& rhs);
      T operator()();
    };
    
    class SCISHARE Variables 
    {
    public:

      Variables         (const string &id, Variables *parent=0);
      virtual           ~Variables();
     
      void              insert(const string &name,
                               const string &value, 
                               const string &type_str = "string",
                               bool propagate = false);
      void              copy_var(const string &from, const string &to);

      bool              exists(const string &varname);
      
      string            get_id();
      int               get_int(const string &);
      double            get_double(const string &);
      bool              get_bool(const string &);
      Color             get_color(const string &);
      string            get_string(const string &);

      enum var_type_e {
        UNKNOWN_E,
        INT_E,
        BOOL_E,
        DOUBLE_E,
        STRING_E,
        COLOR_E
      };
      var_type_e        get_type_e(const string &);
      bool              set_by_string(const string &, const string &);

      struct SCISHARE value_t {
        value_t(string, string, var_type_e);
        bool              update_cache_from_string(Variables *);
        void              update_string_from_cache(Variables *);
        string            name_;
        string            string_value_;
        var_type_e        var_type_;
        int               cache_index_;
        bool              cache_current_;
      };

    private:
      // Disable Copy Constructor
      Variables         (const Variables &copy) { ASSERT(0); }

      void              breakpoint();
      friend class Var<Color>;
      friend class Var<int>;
      friend class Var<bool>;
      friend class Var<string>;
      friend class Var<double>;

      typedef map<string, value_t> name_value_map_t;
      typedef std::set<Variables *> children_t;
      typedef pair<Variables *, value_t *> var_value_t;

      var_value_t               insert_variable(const string &,
                                                var_type_e,
                                                bool);

      var_value_t               find_value_ptr(const string &);

      
      template<class T> 
      int                       set_typed_cache_value(vector<T> &, int &, 
                                                      const T &);

      static var_type_e         string_to_type(string);
      static std::string        type_to_string(var_type_e);
      var_type_e                type_to_enum(int) { return INT_E; }
      var_type_e                type_to_enum(bool){ return BOOL_E; }
      var_type_e                type_to_enum(double){ return DOUBLE_E; }
      var_type_e                type_to_enum(string){ return STRING_E; }
      var_type_e                type_to_enum(Color){ return COLOR_E; }

      void                      set_by_idx(int &, const int &);
      void                      set_by_idx(int &, const bool &);
      void                      set_by_idx(int &, const double &);
      void                      set_by_idx(int &, const std::string &);
      void                      set_by_idx(int &, const Skinner::Color &);

      void                      get_by_idx(int &, int &);
      void                      get_by_idx(int &, bool &);
      void                      get_by_idx(int &, double &);
      void                      get_by_idx(int &, std::string &);
      void                      get_by_idx(int &, Skinner::Color &);

      name_value_map_t          variables_;
      Variables *               parent_;
      children_t                children_;
      std::set<string>          propagate_;
      map<string, string>       alias_;
      
      vector<int>               cached_ints_;
      vector<bool>              cached_bools_;
      vector<double>            cached_doubles_;
      vector<std::string>       cached_strings_;
      vector<Skinner::Color>    cached_colors_;
    };

    template <class T>
    Var<T>::operator T() {
      T temp;
      ASSERT(this->scope_index_  && (*this->scope_index_ >= 0));
      this->scope_->get_by_idx(*this->scope_index_, temp);
      return temp;
    }

    template <class T>
    Var<T> & 
    Var<T>::operator= (const T& rhs) {
      ASSERT(this->scope_);
      ASSERT(this->scope_index_ && *this->scope_index_ >= -1);
      this->scope_->set_by_idx(*this->scope_index_, rhs);
      return *this;
    }

    template <class T>
    Var<T> & 
    Var<T>::operator= (const Var<T>& rhs) {
      this->scope_ = rhs.scope_;
      this->scope_index_ = rhs.scope_index_;
      return *this;
    }

    template <class T>
    Var<T> & 
    Var<T>::operator|= (const Var<T>& rhs) {
      if (this->scope_index_ && 
          (*this->scope_index_ == -1) && 
          (*rhs.scope_index_ != -1)) {
        return operator=(rhs);
      }
      return *this;
    }

    template <class T>
    Var<T> & 
    Var<T>::operator|= (const T& rhs) {
      ASSERT(this->scope_);
      if (this->scope_index_ && (*this->scope_index_ == -1)) {
        this->scope_->set_by_idx(*this->scope_index_, rhs);
      }
      ASSERT(this->scope_index_  && (*this->scope_index_ >= 0));
      return *this;
    }

    template <class T>
    T 
    Var<T>::operator()() {
      T temp;
      ASSERT(this->scope_index_  && (*this->scope_index_ >= 0));
      this->scope_->get_by_idx(*this->scope_index_, temp);
      return temp;
    }

    template<class T>
    int
    Variables::set_typed_cache_value(vector<T> &cache_vector, 
                                     int &index,
                                     const T &typed_value)
    {
      int size = int(cache_vector.size());
      ASSERT(index < size);
      if (index >= 0) {
        cache_vector[index] = typed_value;
      } else {
        index = size;
        cache_vector.push_back(typed_value);
      }      
      return index;
    }


    template <class T>
    Var<T>::Var(Variables *vars, const string &name, const T &init)
    {
      (*this) = Var<T>(vars,name);
      if (!exists()) (*this) = init;
    }


    template <class T>
    Var<T>::Var(Variables *vars, const string &inname) :
      scope_(vars), scope_index_(0)
    {
      string name = inname;
      if (vars->alias_.find(name) != vars->alias_.end()) {
        name = vars->alias_[name];
      }
      Variables::var_value_t varval = vars->find_value_ptr(name);

      Variables::var_type_e var_type = vars->type_to_enum(T());

      if (varval.second) {
        Variables::value_t *value_ptr = varval.second;
        if (value_ptr->var_type_ == Variables::UNKNOWN_E || 
            value_ptr->var_type_ == Variables::STRING_E ) {
          value_ptr->var_type_ = var_type;
        } else if (value_ptr->var_type_ != var_type) {
          throw "invalid type change";        
        }

        if (value_ptr->cache_index_ == -1) {
          value_ptr->update_cache_from_string(varval.first);
        }
      } else {
        varval = vars->insert_variable(name,var_type,false);
      }
      scope_ = varval.first;
      scope_index_ = &varval.second->cache_index_;
    }    

  } // end namespace Skinner
} // end namespace SCIRun


#endif // #define Skinner_Variables_H
