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
    4) Fast evaluation is not quaranteed (see Core::Math:Function documen-
    tation). Also, setting fast evaluation mode required several second of time
    overhead per component and could take significant time sometimes. So there is
    tradeoff between fast and slow kinds of evaluation.

PATTERNS
   
WARNING    
    1) No methods provided for redefinition of domain dimensions.In order
    to redefine it one needs to create new object
    2) Core/CCA/Components are zero-numbered (i.e. for vector 0, 1, 2)

POSSIBLE REVISIONS
    Adding errors handling in case of bad args supplied 
    (log(-1) for instance)
----------------------------------------------------------------------*/
#ifndef SCI_project_GenFunction_h
#define SCI_project_GenFunction_h 1

#include <Core/share/share.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Math/function.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/Assert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Exceptions/DimensionMismatch.h>
#include <Core/Exceptions/InternalError.h>
#include <string>
#include <vector>

namespace SCIRun {
  using std::string;
  
  class GenFunction;
  typedef LockingHandle<GenFunction> GenFunctionHandle;
  
  typedef vector<Function*>   FuncPtrV;
  typedef vector<string>      StrV;
  typedef vector<FuncEvalPtr> FastEvalV;
  typedef vector<double>      RealV;

  class SCICORESHARE GenFunction: public Datatype {
    
    FuncPtrV    d_fns;
    FastEvalV   d_fevals;
    StrV        d_fstr;
    RealV       d_vars;
    double*     d_vars2;

    int     d_domDims;
    int     d_resDims;
    int     d_fnsDims;
    int     d_isFast;
    int     d_isChanged;
   
    //////////
    // Deletes pointers to function in the vector
    void    clear(FuncPtrV&);

    //////////
    // If needed, reset pointers of fast-evaluation
    // /could be long process, for it involves compilation
    // of some source files - see Function.h 
    // for details/
    void    refresh_fast_eval();
 
  public:
    
    //////////
    // Constructors/Destructor
    GenFunction();
    GenFunction(const GenFunction&);
    ~GenFunction();
    GenFunction& operator=(const GenFunction&);
    
    //////////
    // Setting functions
    void set_funcs(const StrV&, int);
    void set_funcs(const FuncPtrV&, int);
    
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
    string    get_str_f(unsigned int) const;

    //////////
    // Evaluation functions
    inline void get_value(double[], int, RealV&) const; 
    
    /////////
    // Getting value of specific component
    inline double get_value(double[], int, int) const;

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
    return d_resDims;
  }
  
  inline int GenFunction::get_dom_dims() const {
    return d_domDims;
  }

  //////////
  // General evaluation function
  inline void GenFunction::get_value(double vrs[], int sz, RealV& out) const{
    ASSERT(d_domDims<=sz);
    out.resize(d_resDims);
      for (int i=0; i<d_resDims; i++)
	out[i]=(d_isFast && d_fevals[i])?(d_fevals[i](vrs)):(d_fns[i]->eval(vrs));
  }

  inline double GenFunction::get_value(double vrs[], int sz, int nc) const{
    ASSERT(d_domDims<=sz && nc < d_resDims);
    return (d_isFast && d_fevals[nc])?(d_fevals[nc](vrs)):(d_fns[nc]->eval(vrs));
  }

} // End namespace SCIRun

#endif
