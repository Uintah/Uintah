/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


//=======================
// function.h
// David Hart
// July 2000
// SCI group
// University of Utah
//=======================

#ifndef __FUNCTION_H__
#define __FUNCTION_H__

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
using namespace std;

class Function;
class FunctionSoLoader;

//----------------------------------------------------------------------

				// fnparsestring parses the function
				// contained in the string "str", and
				// if successful allocates and puts
				// the new "Function*" in the location
				// pointed to by param.
				// i.e. fnparsexxx() should be called
				// like so:

				// Function* f;
				// fnparsestring(equationStr, &f);

				// the caller has the obligation of
				// deleting the returned Function when
				// done with it.

void fnparsestring(const char* str, Function** param);
void fnparsestdin(Function** param);


//----------------------------------------------------------------------
enum ftype {
  
				// special functions
  number, variable,
  
				// unary functions (1 arg)
  exponential, sine, cosine, square, 
  squareroot, logarithm, negation, absolutevalue,
  
				// binary functions (2 args)
  sum, difference, product, quotient, power,

                                // 0 arg functions.
  randomfunction
};

//----------------------------------------------------------------------
typedef double (*FuncEvalPtr)(double* x);

//----------------------------------------------------------------------
class Function {
  
protected:

  ftype		type;

  union {
  double	num;
  
  int		var;		// var==n -> x[n], for example var==0
				// --> x[0], var==1 --> x[1], etc...
  
  Function*	arg1;		// unary or first binary argument
  };
  
  Function*	arg2;		// right binary argument
  
public:

				//=======================
				// constructors

				// create a binary function
  Function(ftype type, Function* arg1, Function* arg2);
				// create a unary function
  Function(ftype type, Function* arg1);
				// create a const number function
  Function(double num);
				// create a function of one variable
  Function(int var);
				// create a copy of a function
  Function(Function* f);

  
				// destructor
  ~Function();

				// return true if f1 == f2;
  friend bool isduplicate(const Function* f1, const Function* f2);

  inline int operator==(const Function& f) const {
    return isduplicate(this, &f);
  }

				// evaluate this function at x
  double eval(double* x) const;

  inline double operator()(double* x) const {
    return eval(x);
  }

				// take partial derivative of this
				// function with respect to variable
				// x[n]
  Function* diff(int n);
  
				// returns true if the calling
				// function is a function of x[n]
  bool functionOf(int n);
                                // returns the order number of last 
                                // input variable
                                // by default they are x0, x1, ...                                
  int  get_dims();
  
  
				// simplify an expression Mostly, we
				// are just checking for
				// simplifications of number cases,
				// like 0 * anything = 0, 0 + anything
				// = anything, etc...
  friend void simplify(Function** f, int printsteps=false);
  friend bool stepsimplify(Function** f);
  
				//=======================
				// output functions
  
				// print a function on the output
				// stream os
  friend ostream& operator<<(ostream& os, const Function* f);
				// print a function to a tex file
  void tex(ostream& texfile);
  

  //=======================
  // getFastEval() gets a pointer to a function that will evaluate
  // this "Function" with the same interface as eval(), but much
  // faster.  A FuncEvalPtr is returned which evalutes the function
  // once (per call) at a point x.  getFastEval() may fail for any
  // number of reasons, and will return a pointer to eval() if this
  // happens.  A call to this function immediately invalidates any
  // function pointers obtained through previous calls to this
  // function.  There is a one or two second overhead to calling
  // getFastEval, so it should be used only if the returned function
  // will be called MANY times.  Otherwise, just calling
  // Function::eval() is probably much faster.
  
  FuncEvalPtr getFastEval();
  
protected:

				// soloader keeps track of all SOs
				// opened by getFastEval() so they can
				// be closed along the way or cleaned
				// up at the end
  static FunctionSoLoader soloader;
};

//----------------------------------------------------------------------
typedef void* sohandle;
struct soentry {
  sohandle handle;
  char* filename;
};
typedef map<Function*, soentry> handleMap;
typedef map<Function*, soentry>::iterator handleIter;

class FunctionSoLoader {
protected:
  handleMap handles;
public:
  FunctionSoLoader() { }
  ~FunctionSoLoader();
  sohandle getHandle(Function* f, char* soFileName);
  void deleteHandle(Function*);
  void deleteHandle(handleIter);
};
  
} // End namespace SCIRun

#endif // __FUNCTION_H__

















