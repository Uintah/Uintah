/*----------------------------------------------------------------------
CLASS
    GenFunction

    Function defined on some arbitrary domain and of arbitrary dimensions

GENERAL INFORMATION
    Created by:
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    August 2000
    
    Copyright (C) 2000 SCI Group

KEYWORDS
   

DESCRIPTION
    
    Invariants and behavior:
    1) There is always at least one defined component of the function
    2) Domain dimensions could be defined only at construction
    3) Number of arguments supplied to evaluation functions should be more
    or equal to dom_dims value; otherwise DimensionMismatch(int, int) is
    thrown.
    4) Fast evaluation is not quaranteed (see SCICore::Math:Function documen-
    tation). Also, setting fast evaluation mode required several second of time
    overhead per component and could take significant time sometimes. So there is
    tradeoff between fast and slow kinds of evaluation.

PATTERNS
   
WARNING    
    No methods provided for redefinition of domain dimensions.In order
    to redefine it one needs to create new object
    
POSSIBLE REVISIONS
    Adding errors handling in case of bad args supplied 
    (log(-1) for instance)
----------------------------------------------------------------------*/
#ifndef SCI_project_GenFunction_h
#define SCI_project_GenFunction_h 1

#include <SCICore/share/share.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Math/function.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Exceptions/DimensionMismatch.h>
#include <SCICore/Exceptions/InternalError.h>
#include <string>
#include <vector>

namespace SCICore{
namespace Datatypes{
  
  using SCICore::Geometry::Vector;
  using SCICore::PersistentSpace::Piostream;
  using SCICore::PersistentSpace::PersistentTypeID;
  using SCICore::Containers::LockingHandle;

  using namespace SCICore::Math;
  using namespace SCICore::Exceptions;

  using std::string;
  
  class GenFunction;
  typedef LockingHandle<GenFunction> GenFunctionHandle;
  
  typedef vector<Function*>   FuncPtrV;
  typedef vector<string>      StrV;
  typedef vector<FuncEvalPtr> FastEvalV;
  typedef vector<double>      RealV;

  class SCICORESHARE GenFunction: public Datatype {
    FuncPtrV    fns;
    FastEvalV   fevals;
    StrV        fstr;
    RealV       vars;
    double*     vars2;

    int     dom_dims;
    int     res_dims;
    int     fns_dims;
    int     is_fast;
    int     is_changed;
 
    void    clear(FuncPtrV&);
    void    refresh_fast_eval();
 
  public:
    
    //////////
    // Constructors/Destructor
    GenFunction(const FuncPtrV&, int);
    GenFunction(const StrV&, int);
    GenFunction(const GenFunction&);
    ~GenFunction();
    GenFunction& operator=(const GenFunction&);
    
    //////////
    // Setting functions
    void set_funcs(const StrV&);
    void set_funcs(const FuncPtrV&);
    
    //////////
    // Dimensions of function definition domain
    inline int get_dom_dims() const;

    //////////
    // Number of components
    inline int  get_num_comp() const;
    void set_num_comp(int);

    //////////
    // Clones function object of the specified component
    void get_component(int, Function**);

    //////////
    // Returns string representation of the component
    string    get_str_f(int) const;

    //////////
    // Evaluation functions
    inline void get_value(const RealV&, RealV&) const;
    inline void get_value(double[], int, RealV&) const;
    inline double get_first_val(double[], int) const;

    //////////
    // Set fast/slow evaluation modes
    // the success depends on if we managed to get FasfEvalPtr from Function()
    void    set_speed(bool);
    
    //////////
    // Persistent representation...
    // No function object is carried around while serializing
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
  };
 
  inline int GenFunction::get_num_comp() const{
    return res_dims;
  }
  
  inline int GenFunction::get_dom_dims() const {
    return dom_dims;
  }

  //////////
  // Most general evaluation function
  inline void GenFunction::get_value(const RealV& in, RealV& out) const{
    int insz=in.size();
    if (dom_dims>insz)
      throw DimensionMismatch(insz, dom_dims);
    else {
      NOT_FINISHED("GenFunction::get_value() needs optimization");
      for (int i=0; i<dom_dims; i++)
	vars2[i]=in[i];
      for (int i=0; i<res_dims; i++)
	out[i]=(is_fast && fevals[i])?(fevals[i](vars2)):(fns[i]->eval(vars2));
    }
  }
  
  //////////
  // the same as previous but gets array of doubles inctead of vector
  inline void GenFunction::get_value(double vrs[], int sz, RealV& out) const{
    if (dom_dims>sz)
      throw DimensionMismatch(sz, dom_dims);
    else {
      NOT_FINISHED("GenFunction::get_value() needs optimization");
      out.resize(res_dims);
      for (int i=0; i<res_dims; i++)
	out[i]=(is_fast && fevals[i])?(fevals[i](vrs)):(fns[i]->eval(vrs));
    }
  }

  //////////
  // evaluation of the first, but often the last component of the function  
  inline double GenFunction::get_first_val(double vrs[], int d) const{
    if (d<dom_dims)
      throw DimensionMismatch(d, dom_dims);
    return (is_fast && fevals[0])?(fevals[0](vrs)):(fns[0]->eval(vrs));
  }

}
}

#endif

























































































































































































































































































