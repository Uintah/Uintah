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
// function.cc
// David Hart
// July 2000
// SCI group
// University of Utah
//=======================

#include <Core/Math/function.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <dlfcn.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
using namespace std;

namespace SCIRun {

//----------------------------------------------------------------------
FunctionSoLoader Function::soloader;

//----------------------------------------------------------------------
Function::Function(ftype type, Function* arg1, Function* arg2):
  type(type), arg1(arg1), arg2(arg2)
{
}
  
//----------------------------------------------------------------------
Function::Function(ftype type, Function* arg1):
  type(type), arg1(arg1)
{
  arg2 = NULL;
}
  
//----------------------------------------------------------------------
Function::Function(double num): num(num) {
  type = number;
  arg2 = NULL;
}
	
//----------------------------------------------------------------------
Function::Function(int var): var(var) {
  type = variable;
  arg2 = NULL;
}
  
//----------------------------------------------------------------------
Function::Function(Function* f) {
  type = f->type;
  arg1 = arg2 = NULL;
  if (type == number) num = f->num;
  else if (type == variable) var = f->var;
  else if (f->arg1) arg1 = new Function(f->arg1);
  if (f->arg2) arg2 = new Function(f->arg2);
}
  
//----------------------------------------------------------------------
Function::~Function() {
  if (type != number && type != variable) {
    delete arg1;
    delete arg2;		// okay to delete NULL
  }
  soloader.deleteHandle(this);
}
  
//----------------------------------------------------------------------
bool Function::functionOf(int n) {
  switch(type) {
      
  case variable:
    if (var == n) return true;
    else return false;
      
  case number:
    return false;
      
  case negation: case exponential: case sine: case cosine: case square:
  case squareroot: case logarithm: case absolutevalue:
    return arg1->functionOf(n);
      
  case sum: case difference: case product: case quotient: case power:
    return (arg1->functionOf(n) || arg2->functionOf(n));
      
  default:
    return false;
      
  }
}

int Function::get_dims() {
  
  switch(type) {    

  case variable:
    return var+1;

  case number:
    return 0;
      
  case negation: case exponential: case sine: case cosine: case square:
  case squareroot: case logarithm: case absolutevalue:
    return arg1->get_dims();
      
  case sum: case difference: case product: case quotient: case power:
    {
      int d1=arg1->get_dims();
      int d2=arg2->get_dims();
      return (d1>d2)?d1:d2;
    }
  
  default:
    return 0;
  }
}

//----------------------------------------------------------------------
				// evaluate this function at x
double Function::eval(double* x) const {
  double temp;
  switch(type) {
    
  case number:		return num;
      
  case variable:	return x[var];
    
  case negation:	return -arg1->eval(x);
  case exponential:	return exp(arg1->eval(x));
  case sine:		return sin(arg1->eval(x));
  case cosine:		return cos(arg1->eval(x));
  case logarithm:	return log(arg1->eval(x));
  case absolutevalue:	return fabs(arg1->eval(x));
  case square:		temp = arg1->eval(x); return temp*temp;
  case squareroot:	return sqrt(arg1->eval(x));
      
  case sum:		return (arg1->eval(x) + arg2->eval(x));
  case difference:	return (arg1->eval(x) - arg2->eval(x));
  case product:		return (arg1->eval(x) * arg2->eval(x));
  case quotient:	return (arg1->eval(x) / arg2->eval(x));
  case power:		return pow(arg1->eval(x), arg2->eval(x));

  case randomfunction:	return drand48();
    
  default:
    cerr << "Function::eval() error: "
	 << "bad function type "
	 << type << endl;
  }
  return 0.0;
}
  
//----------------------------------------------------------------------
				// take partial derivative of this
				// function with respect to variable
Function* Function::diff(int n) {
  switch(type) {

				//=======================
				// dx(x) = 1
  case variable:
    if (n == var) {
      return new Function(1.0);
    }
				//=======================
				// dx(y) = 0
    else {
      return new Function(0.0);
    }

				//=======================
				// dx(n) = 0
  case number:
    return new Function(0.0);

				//=======================
				// dx(-u) = -dx(u)
  case negation:
    return new Function(negation, arg1->diff(n));

				//=======================
				// dx(e^u) = dx(u)*e^u
  case exponential:
    return new Function(product, arg1->diff(n), new Function(this));

				//=======================
				// dx(ln(u)) = dx(u)/u
  case logarithm:
    return new Function(quotient, arg1->diff(n), new Function(arg1));

				//=======================
				// dx(sin(u)) = dx(u)*cos(u)
  case sine:
    return new Function(product, arg1->diff(n), new Function(cosine,
      new Function(arg1)));

				//=======================
				// dx(cos(u)) = -dx(u)*cos(u)
  case cosine:
    return new Function(negation, new Function(product, arg1->diff(n), 
      new Function(sine, new Function(arg1))));

				//=======================
				// dx(u^2) = 2*u*dx(u)
  case square:
    return new Function(product, new Function(2.0), new Function(product, 
      new Function(arg1), arg1->diff(n)));

				//=======================
				// dx(u^(1/2)) = dx(u)/(2*u^(1/2))
  case squareroot:
    return new Function(quotient, arg1->diff(n), new Function(product, 
      new Function(2.0), new Function(this)));

				//=======================
				// dx(|u|) = |u|*dx(u)/u
  case absolutevalue:
    return new Function(quotient, new Function(product,
      new Function(absolutevalue, new Function(arg1)),
      arg1->diff(n)), new Function(arg1));

				//=======================
				// dx(u+v) = dx(u)+dx(v)
  case sum:
    return new Function(sum, arg1->diff(n), arg2->diff(n));

				//=======================
				// dx(u-v) = dx(u)-dx(v)
  case difference:
    return new Function(difference, arg1->diff(n), arg2->diff(n));

				//=======================
				// dx(u*v) = dx(u)*v + u*dx(v)
  case product:
    return new Function(sum, new Function(product, arg1->diff(n),
      new Function(arg2)), new Function(product,
	new Function(arg1), arg2->diff(n)));

				//=======================
				// dx(u/v) = dx(u)*v+u*dx(v)/v^2
  case quotient:
    return new Function(quotient, new Function(sum,
      new Function(product, arg1->diff(n), 
	new Function(arg2)), new Function(product,
	  new Function(arg1), arg2->diff(n))), new Function(square,
	    new Function(arg2)));

				   
 				//=======================
				// dx(u^v) = (v*dx(u)/u+dx(v)*ln(u))*(u^v)
  case power:
    return new Function(product, new Function(sum,
      new Function(product, new Function(arg2),
	new Function(quotient, arg1->diff(n), new Function(arg1))),
      new Function(product, arg2->diff(n),
	new Function(logarithm, new Function(arg1)))),
      new Function(this));
    
				//=======================
  default:
    cerr << "Function::diff() errror: "
	 << "bad function type "
	 << type << endl;
  }
  return NULL;
}
  
//----------------------------------------------------------------------
bool isduplicate(const Function* f1, const Function* f2) {
  if (f1->type != f2->type) {
    return false;
  }

  switch(f1->type) {
				// special functions
  case variable:
    return (f1->var == f2->var);

  case number:
    return (f1->num == f2->num);
		
				// functions of one argument
  case exponential: case sine: case cosine: case square: 
  case squareroot: case logarithm: case negation: case absolutevalue:
    return isduplicate(f1->arg1, f2->arg1);
	
				// functions of two arguments
  case sum: case difference: case product: case quotient: case power:
    return (isduplicate(f1->arg1, f2->arg1) &&
      isduplicate(f1->arg2, f2->arg2));

  case randomfunction:
    return false;
  }

  return false;
}


//----------------------------------------------------------------------
void simplify(Function** f, int printsteps) {
  while (stepsimplify(f)) {
    if (printsteps) cout << "simp:" << *f << endl;
  }
}

//----------------------------------------------------------------------

				// (try to) simplify an expression to
				// make printing prettier, or speedup
				// evaluation.  We are mostly just
				// checking for simplifications of
				// number cases, like 0 * x = 0, 0 + x
				// = x, etc...

bool stepsimplify(Function** f) {
  Function* temp;
#define FINISH(x) temp = (x); delete (*f); (*f) = temp; return true
    
  switch((*f)->type) {

				//=======================
				// variables and numbers can't be
				// simplified
  case variable: case number:
    return false;

				//=======================
				// unary functions
  case exponential: case sine: case cosine: case square:
  case squareroot: case negation: case logarithm: case absolutevalue:

				// only do one step at a time
    if (stepsimplify(&((*f)->arg1))) return true;
    
				// a function of a number is a number
    if ((*f)->arg1->type == number) {
      FINISH(new Function((*f)->eval(NULL)));
    }
				// a double negative is a positive
    else if ((*f)->type == negation && (*f)->arg1->type == negation) {
      FINISH(new Function((*f)->arg1->arg1));
    }

    break;
    
				//=======================
				// binary functions
  case sum: case difference: case product: case quotient: case power:

				// only do one step at time
    if (stepsimplify(&((*f)->arg1)) || stepsimplify(&((*f)->arg2))) {
      return true;
    }

				// a function of two numbers is a
				// number...
    if ((*f)->arg1->type == number && (*f)->arg2->type == number) {
      FINISH(new Function((*f)->eval(NULL)));
    }
				// search for products with 0, sum
				// with 0, product with 1, etc..
    else if ((*f)->arg1->type == number) {
      if ((*f)->arg1->num == 0.0) {
	switch((*f)->type) {
	case sum:	FINISH(new Function((*f)->arg2));
	case difference: FINISH(new Function(negation,
	  new Function((*f)->arg2)));
	case power:	FINISH(new Function((*f)->arg1->num));
	case product:	FINISH(new Function(0.0));
	case quotient:	FINISH(new Function(0.0));
	default: throw "bad enum";
	}
      }
      else if ((*f)->arg1->num == 1.0) {
	switch((*f)->type) {
	case product:	FINISH(new Function((*f)->arg2));
	case power:	FINISH(new Function(1.0));
	default: throw "bad enum";
	}
      }
    }
    else if ((*f)->arg2->type == number) {
      if ((*f)->arg2->num == 0.0) {
	switch((*f)->type) {
	case sum:        FINISH(new Function((*f)->arg1));
	case difference: FINISH(new Function((*f)->arg1));
	case product:    FINISH(new Function(0.0));
	case power:      FINISH(new Function(1.0));
	case quotient:
	  cerr << "Function::simplify() warning: "
	       << "division by zero\n";
	  return false;
	default: throw "bad enum";
	}
      }
      else if ((*f)->arg2->num == 1.0 && (((*f)->type == product) ||
	((*f)->type == quotient) || ((*f)->type == power)))
	{
	  FINISH(new Function((*f)->arg1));
	}
    }

    if ((*f)->type == product) {
				// negation propagates outward
      if ((*f)->arg1->type == negation) {
	FINISH(new Function(negation, new Function(product,
	  new Function((*f)->arg1->arg1),
	  new Function((*f)->arg2))));
      }
				// negation propagates outward
      if ((*f)->arg2->type == negation) {
	FINISH(new Function(negation, new Function(product,
	  new Function((*f)->arg1),
	  new Function((*f)->arg2->arg1))));
      }
				// coefficients propagate to the front
      if ((*f)->arg2->type == number &&
	(*f)->arg1->type != number) {
	FINISH(new Function(product, new Function((*f)->arg2),
	  new Function((*f)->arg1)));
      }

				// collapse a product of two numbers
      if ((*f)->arg1->type == number &&
	(*f)->arg2->type == product &&
	(*f)->arg2->arg1->type == number)
	{
	  FINISH(new Function(product,
	    new Function((*f)->arg1->num *
	      (*f)->arg2->arg1->num),
	    new Function((*f)->arg2->arg2)));
	}
	
				// coefficients propagate outward
      if ((*f)->arg1->type == product &&
	(*f)->arg1->arg1->type == number)
	{
	  FINISH(new Function(product,
	    new Function((*f)->arg1->arg1->num),
	    new Function(product,
	      new Function((*f)->arg1->arg2),
	      new Function((*f)->arg2))));
	}
	  
				// coefficients propagate outward
      if ((*f)->arg2->type == product &&
	(*f)->arg2->arg1->type == number)
	{
	  FINISH(new Function(product,
	    new Function((*f)->arg2->arg1->num),
	    new Function(product,
	      new Function((*f)->arg1),
	      new Function((*f)->arg2->arg2))));
	}

				// look for a product simplifying to
				// a square
      if (isduplicate((*f)->arg1, (*f)->arg2)) {
	FINISH(new Function(square, new Function((*f)->arg1)));
      }

	
    }

    if ((*f)->type == quotient) {
				// negation propagates outward
      if ((*f)->arg1->type == negation) {
	FINISH(new Function(negation, new Function(quotient,
	  new Function((*f)->arg1->arg1),
	  new Function((*f)->arg2))));
      }
				// negation propagates outward
      if ((*f)->arg2->type == negation) {
	FINISH(new Function(negation, new Function(quotient,
	  new Function((*f)->arg1),
	  new Function((*f)->arg2->arg1))));
      }

				// coefficients propagate to the front
      if ((*f)->arg1->type == number &&
	(*f)->arg1->num != 1.0)
	{
	  FINISH(new Function(product, new Function((*f)->arg1->num),
	    new Function(quotient,
	      new Function(1.0),
	      new Function((*f)->arg2))));
	}

				// coefficients propagate to the front
      if ((*f)->arg2->type == number &&
	(*f)->arg1->type != number)
	{
	  FINISH(new Function(product, new Function(1.0 / (*f)->arg2->num),
	    new Function((*f)->arg1)));
	}

				// simplify in the case of two numbers
      if ((*f)->arg1->type == number &&
	(*f)->arg2->type == product &&
	(*f)->arg2->arg1->type == number)
	{
	  FINISH(new Function(product,
	    new Function((*f)->arg1->num /
	      (*f)->arg2->arg1->num),
	    new Function(quotient,
	      new Function(1.0),
	      new Function((*f)->arg2->arg2))));
	}
	
				// coefficients propagate outward
      if ((*f)->arg1->type == product &&
	(*f)->arg1->arg1->type == number)
	{
	  FINISH(new Function(product,
	    new Function((*f)->arg1->arg1->num),
	    new Function(quotient,
	      new Function((*f)->arg1->arg2),
	      new Function((*f)->arg2))));
	}
	
				// coefficients propagate outward
      if ((*f)->arg2->type == product &&
	(*f)->arg2->arg1->type == number)
	{
	  FINISH(new Function(product,
	    new Function(1.0 / (*f)->arg2->arg1->num),
	    new Function(quotient,
	      new Function((*f)->arg1),
	      new Function((*f)->arg2->arg2))));
	}

				// look for a quotient simplifying to
				// one.
      if (isduplicate((*f)->arg1, (*f)->arg2)) {
	FINISH(new Function(1.0));
      }
	
    }
      
				// optimize negation
    if ((*f)->type == sum) {
				// cancel a double negative
      if ((*f)->arg1->type == negation &&
	(*f)->arg2->type == negation)
	{
	  FINISH(new Function(negation, new Function(sum,
	    new Function((*f)->arg1->arg1),
	    new Function((*f)->arg2->arg1))));
	}
				// prettier ordering
				// -a+b -> b-a
      if ((*f)->arg1->type == negation) {
	FINISH(new Function(difference,
	  new Function((*f)->arg2),
	  new Function((*f)->arg1->arg1)));
      }
				// optimize out a minus
				// a + -b -> a-b
      if ((*f)->arg2->type == negation) {
	FINISH(new Function(difference,
	  new Function((*f)->arg1),
	  new Function((*f)->arg2->arg1)));
      }
				// look for sum simplifying to 2*
      if (isduplicate((*f)->arg1, (*f)->arg2)) {
	FINISH(new Function(product,
	  new Function(2.0),
	  new Function((*f)->arg1)));
      }
				// coefficients propagate to the back
      if ((*f)->arg1->type == number &&
	(*f)->arg2->type != number) {
	FINISH(new Function(sum,
	  new Function((*f)->arg2),
	  new Function((*f)->arg1)));
      }
				// coefficients group together
      if ((*f)->arg2->type == number &&
	(*f)->arg1->type == sum &&
	(*f)->arg1->arg2->type == number)
	{
	  FINISH(new Function(sum,
	    new Function((*f)->arg1->arg1),
	    new Function((*f)->arg1->arg2->num + (*f)->arg2->num)));
	}
      
    }

    break;
      
  default:
    cerr << "Function::simplify() error: "
	 << "bad function type " << ((*f)->type) << endl;
    return false;
  }

				// if we arrived down here, we can't
				// simplify anymore
  return false;
}

//----------------------------------------------------------------------
				// print a function on the output
				// stream os
ostream& operator<<(ostream& os, const Function* f) {
  switch(f->type) {
      
  case variable:
    os << "x[" << f->var << "]";
    break;
      
  case number:
    os << f->num;
    break;
      
  case sum: case difference: case product: case quotient:
    os << "(" << f->arg1;
    switch(f->type) {
    case sum:		os << " + "; break;
    case difference:	os << " - "; break;
    case product:	os << " * "; break;
    case quotient:	os << " / "; break;
    default: throw "bad enum";
    }
    os << f->arg2 << ")";
    break;

  case power:
    os << "pow(" << f->arg1 << ", " << f->arg2 << ")";
    break;
      
  case sine: case cosine: case logarithm: case exponential:
  case negation: case square: case absolutevalue: case squareroot:
    switch(f->type) {
    case sine:		os << "sin";	break;
    case cosine:	os << "cos";	break;
    case logarithm:	os << "log";	break;
    case exponential:	os << "exp";	break;
    case negation:	os << "-";	break;
    case squareroot:	os << "sqrt";	break;
    case square:	os << "sqr";	break;
    case absolutevalue:	os << "abs";	break;
    default: throw "bad enum";
    }
    os << "(" << f->arg1 << ")";
    break;
            
  default:
    cerr << "Function::operator<<() error: "
	 << "bad function type " << f->type << endl;
    break;
  }
  return os;
}
    
//----------------------------------------------------------------------
				// print a function to a tex file
void Function::tex(ostream& texfile) {
  int parens = false;
  int parens2 = false;
    
  switch(type) {
      
  case variable:
    texfile << "x_" << var;
    break;
      
  case number:
    texfile << num;
    break;
      
  case sum:
  case difference:
    //texfile << "\\left(";
    arg1->tex(texfile);
    switch(type) {
    case sum: texfile << "+"; break;
    case difference: texfile << "-"; break;
    default: throw "bad enum";
    }
    if (arg2->type == sum || arg2->type == difference) {
      texfile << "\\left(";
    }
    arg2->tex(texfile);
    if (arg2->type == sum || arg2->type == difference) {
      texfile << "\\right)";
    }
    //texfile << "\\right)";
    break;
      
  case product:
    parens = (arg1->type == sum || arg1->type == negation ||
      arg1->type == difference);
    parens2 = (arg2->type == sum || arg2->type == negation ||
      arg2->type == difference);
    if (parens) texfile << "\\left(";
    arg1->tex(texfile);
    if (parens) texfile << "\\right)";      
    texfile << " ";
    if (parens2) texfile << "\\left(";
    arg2->tex(texfile);
    if (parens2) texfile << "\\right)";
    break;

  case quotient:
    texfile << "\\frac{";
    arg1->tex(texfile);
    texfile << "}{";
    arg2->tex(texfile);
    texfile << "}";
    break;
      
  case power:
  case square:
    if (arg1->type != number &&
      arg1->type != variable &&
      arg1->type != absolutevalue &&
      arg1->type != squareroot) {
      parens = true;
    }
    if (parens) texfile << "\\left(";
    arg1->tex(texfile);
    if (parens) texfile << "\\right)";
    texfile << "^";
    if (type == power) {
      texfile << "{";
      arg2->tex(texfile);
      texfile << "}";
    }
    else {
      texfile << "2";
    }
    break;
      
  case negation:
    texfile << "-";
    if (arg1->type == sum || arg1->type == negation)
      texfile << "\\left(";
    arg1->tex(texfile);
    if (arg1->type == sum || arg1->type == negation)
      texfile << "\\right)";      
    break;
      
  case sine: case cosine: case logarithm: case exponential:
    switch(type) {
    case sine:		texfile << "sin";	break;
    case cosine:	texfile << "cos";	break;
    case logarithm:	texfile << "ln";	break;
    case exponential:	texfile << "e^";	break;
    default: throw "bad enum";
    }
    if (type == exponential) texfile << "{";
    else texfile << "\\left(";
    arg1->tex(texfile);
    if (type == exponential) texfile << "}";
    else texfile << "\\right)";
    break;
      
  case squareroot:
    texfile << "\\sqrt{";
    arg1->tex(texfile);
    texfile << "}";
    break;

  case absolutevalue:
    texfile << "\\left|";
    arg1->tex(texfile);
    texfile << "\\right|";
    break;
      
  default:
    cerr << "Function::tex() error: "
	 << "bad function type " << type << endl;
    break;
  }

}

//----------------------------------------------------------------------
FuncEvalPtr Function::getFastEval() {
				// unlikely-to-conflict-with-anything
				// temp filename
  char* fname = "/tmp/tmp_eq_%.4d.cc";
  char filename[256];

				// attempt to open the temp code file
  sprintf(filename, fname, (rand()+getpid()) % 9999);
  ofstream ofile(filename, ios::out);
  if (!ofile) {
    cerr << "Function::getFastEval() error: "
	 << "couldn't write temp file\n";
    return NULL;
  }

				// write our function to the code file
  ofile
    << "\n"
    << "/* delete me */\n"
    << "\n"
    << "#include<math.h>\n"
    << "\n"
    << "#define abs fabs\n"
    << "\n"
    << "extern \"C\" {\n"
    << "\n"
    << "  inline double sqr(double x) { return x*x; }\n"
    << "\n"
    << "  double func_getfasteval(double* x) {\n"
    << "    return (" << this << ");\n"
    << "  }\n"
    << "\n"
    << "}\n";
  ofile.close();

				// attempt to compile the code file we
				// just wrote into a shared object
				// file
  char command[1024];
  sprintf(command, "CC -O -shared %s -o %s.so", filename, filename);
  if (system(command) != 0) {
    cerr << "Function::getFastEval() error: "
	 << "system call failed:\n" << command << endl;
    return NULL;
  }

				// remove the temp code file
  sprintf(command, "rm -f %s", filename);
  system(command);

				// attepmt to load and link the
				// function we just compiled
  char* sofilename = new char[strlen(filename)+4];
  sprintf(sofilename, "%s.so", filename);
  
  sohandle handle = soloader.getHandle(this, sofilename);
  if (handle == NULL) return NULL;

				// return the function pointer
  FuncEvalPtr f = (FuncEvalPtr)dlsym(handle, "func_getfasteval");
  if (f == NULL) {
    cerr << "Function::getFastEval() error: "
	 << "dlsym failed" << endl;
    return NULL;
  }
  return f;
}

//----------------------------------------------------------------------
FunctionSoLoader::~FunctionSoLoader() {
				// close all SOs that are still open
  handleIter it;
  for (it = handles.begin(); it != handles.end(); ++it) {
    deleteHandle(it);
  }
}

//----------------------------------------------------------------------
sohandle
FunctionSoLoader::getHandle(Function* f, char* soFileName) {
  sohandle h;
				// if this Function already had an
				// SO, close it before we make a new
				// one
  handleIter it = handles.find(f);
  if (it != handles.end()) {
    deleteHandle(it);
  }
  h = dlopen(soFileName, RTLD_NOW);
  if (h == NULL) {
    cerr << "SoLoader::getHandle() error:"
	 << " dlopen failed" << endl;
  }
  else {
    soentry s;
    s.handle = h;
    s.filename = soFileName;
    handles[f] = s; // record this SO in the map
  }
  return h;
}

//----------------------------------------------------------------------
void
FunctionSoLoader::deleteHandle(Function* f) {
  handleIter it = handles.find(f);
  if (it != handles.end()) deleteHandle(it);
}

//----------------------------------------------------------------------
void
FunctionSoLoader::deleteHandle(handleIter it) {
				// close the so file
  dlclose((*it).second.handle);
  char command[256];
				// erase the so file
  snprintf(command, 255, "rm -f %s", (*it).second.filename);
  system(command);

  				// remove the soentry from the map.
  delete [] (*it).second.filename;
  handles.erase(it);
}

} // End namespace SCIRun
