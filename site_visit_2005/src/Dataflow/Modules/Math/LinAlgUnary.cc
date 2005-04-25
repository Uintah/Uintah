/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

/*
 *  LinAlgUnary: Unary matrix operations -- just transpose for now
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Math/LinAlgUnary.h>
#include <Core/Containers/HashTable.h>
#include <iostream>
#include <sstream>
#include <math.h>

namespace SCIRun {

class LinAlgUnary : public Module {
  GuiString op_;
  GuiString function_;
  void insertion_sort(double *x, int n);
  void subtract_mean(double *x, int n);
  void normalize(double *x, int n);
  void round(double *x, int n);
  void Floor(double *x, int n);
  void Ceil(double *x, int n);
public:
  LinAlgUnary(GuiContext* ctx);
  virtual ~LinAlgUnary();
  virtual void execute();
};

DECLARE_MAKER(LinAlgUnary)
LinAlgUnary::LinAlgUnary(GuiContext* ctx)
: Module("LinAlgUnary", ctx, Filter,"Math", "SCIRun"),
  op_(ctx->subVar("op")), function_(ctx->subVar("function"))
{
}

LinAlgUnary::~LinAlgUnary()
{
}

void LinAlgUnary::insertion_sort(double *x, int n) {
  double tmp;
  for (int i=0; i<n-1; i++)
    for (int j=i+1; j<n; j++)
      if (x[i] > x[j]) {
	tmp = x[i]; x[i]=x[j]; x[j]=tmp;
      }
}

void LinAlgUnary::subtract_mean(double *x, int n) {
  double sum = 0.0;
  for (int i=0; i<n; i++) {
    sum = sum + x[i];
  }
  double avg = sum / (double)n;
  for (int i=0; i<n; i++) {
    x[i] = x[i] - avg;
  }
}

void LinAlgUnary::normalize(double *x, int n) {
  double min =  1.0e36;
  double max = -1.0e36;
  for (int i=0; i<n; i++) {
    if( min > x[i] ) min = x[i];
    else if( max < x[i] ) max = x[i];
  }
  double mult = 1.0 / (max-min);
  for (int i=0; i<n; i++) {
    x[i] = (int) ((x[i]-min) * mult);
  }
}

void LinAlgUnary::round(double *x, int n) {
  for (int i=0; i<n; i++) {
    x[i] = (int) (x[i] + (x[i] < 0 ?  -0.5 : +0.5) );
  }
}

void LinAlgUnary::Floor(double *x, int n) {
  for (int i=0; i<n; i++) {
    x[i] = floor(x[i]);
  }
}

void LinAlgUnary::Ceil(double *x, int n) {
  for (int i=0; i<n; i++) {
    x[i] = ceil(x[i]);
  }
}

void LinAlgUnary::execute() {
  MatrixIPort* imat_ = (MatrixIPort *)get_iport("Input");
  MatrixOPort* omat_ = (MatrixOPort *)get_oport("Output");

  update_state(NeedData);
  MatrixHandle mh;
  if (!imat_->get(mh) || !mh.get_rep()) {
    error( "No handle or representation" );
    return;
  }

  string op = op_.get();

  MatrixHandle m;

  if (op == "Transpose") {
    Matrix *mat = mh->transpose();
    m = mat;
  } else if (op == "Invert") {
    DenseMatrix *dm = mh->dense();
    if (! dm->invert()) {
      error("Input Matrix not invertible.");
      return;
    }
    m = dm;
  } else if (op == "Sort") {
    // Not real efficient for sparse row matrices, but works now.
    m = (mh->is_sparse())?mh->dense():mh->clone();
    insertion_sort(m->get_data_pointer(), m->get_data_size());
    if (mh->is_sparse()) { m = m->sparse(); }
  } else if (op == "Subtract_Mean") {
    m = (mh->is_sparse())?mh->dense():mh->clone();
    subtract_mean(m->get_data_pointer(), m->get_data_size());
  } else if (op == "Normalize") {
    m = mh->clone();
    normalize(m->get_data_pointer(), m->get_data_size());
  } else if (op == "Round") {
    m = mh->clone();
    round(m->get_data_pointer(), m->get_data_size());
  } else if (op == "Floor") {
    m = mh->clone();
    Floor(m->get_data_pointer(), m->get_data_size());
  } else if (op == "Ceil") {
    m = mh->clone();
    Ceil(m->get_data_pointer(), m->get_data_size());
  } else if (op == "Function") {
    if (mh->is_sparse())
    {
      remark("Only calling function for non-zero sparse matrix elements.");
    }

    // Remove trailing white-space from the function string.
    string func = function_.get();
    while (func.size() && isspace(func[func.size()-1]))
    {
      func.resize(func.size()-1);
    }

    // Compile the function.
    int hoffset = 0;
    Handle<LinAlgUnaryAlgo> algo;
    for( ;; )
    {
      CompileInfoHandle ci =
	LinAlgUnaryAlgo::get_compile_info(func, hoffset);
      if (!DynamicCompilation::compile(ci, algo, false, this))
      {
	error("Your function would not compile.");
	gui->eval(id + " compile_error "+ci->filename_);
	DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
	return;
      }
      if (algo->identify() == func)
      {
	break;
      }
      hoffset++;
    }

    // Get the data from the matrix, iterate over it calling the function.
    m = mh->clone();
    double *x = m->get_data_pointer();
    const unsigned int n = m->get_data_size();
    for (unsigned int i = 0; i < n; i++)
    {
      x[i] = algo->user_function(x[i]);
    }
  } else {
    warning("Don't know operation "+op);
    return;
  }

  omat_->send(MatrixHandle(m));
}


CompileInfoHandle
LinAlgUnaryAlgo::get_compile_info(const string &function,
				  int hashoffset)

{
  unsigned int hashval = Hash(function, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("LinAlgUnaryInstance" + to_string(hashval));
  static const string base_class_name("LinAlgUnaryAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + ".",
		       base_class_name,
		       template_name + ";//",
		       "");

  // Code for the function.
  string class_declaration =
    string("using namespace SCIRun;\n\n") + 
    "class " + template_name + " : public LinAlgUnaryAlgo\n" +
    "{\n" +
    "  virtual double user_function(double x)\n" +
    "  {\n" +
    "    return (" + function + ");\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + string_Cify(function) + "\"); }\n" +
    "};\n";

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_post_include(class_declaration);
  return rval;
}


} // End namespace SCIRun
