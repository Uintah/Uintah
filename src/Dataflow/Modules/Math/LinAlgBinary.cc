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
 *  LinAlgBinary: Binary matrix operations -- add, multiply
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/MatrixOperations.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/HashTable.h>
#include <Dataflow/Modules/Math/LinAlgBinary.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class LinAlgBinary : public Module {
  MatrixIPort* imatA_;
  MatrixIPort* imatB_;
  MatrixOPort* omat_;

  GuiString op_;
  GuiString function_;
public:
  LinAlgBinary(GuiContext* ctx);
  virtual ~LinAlgBinary();
  virtual void execute();
};

DECLARE_MAKER(LinAlgBinary)
LinAlgBinary::LinAlgBinary(GuiContext* ctx)
: Module("LinAlgBinary", ctx, Filter,"Math", "SCIRun"),
  op_(ctx->subVar("op")), function_(ctx->subVar("function"))
{
}

LinAlgBinary::~LinAlgBinary()
{
}

void LinAlgBinary::execute() {
  imatA_ = (MatrixIPort *)get_iport("A");
  imatB_ = (MatrixIPort *)get_iport("B");
  omat_ = (MatrixOPort *)get_oport("Output");
  
  update_state(NeedData);
  MatrixHandle aH, bH;
  if (!imatA_->get(aH)) {
    if (!imatB_->get(bH))
      error( "No handle or representation" );
      return;
  } else imatB_->get(bH);
      
  if (!aH.get_rep()) {
    warning("Empty input matrix A.");
  }
  if (!bH.get_rep()) {
    warning("Empty input matrix B.");
  }

  string op = op_.get();
  if (op == "Add") {
    if (!aH.get_rep()) {
      error("Empty A matrix for Add");
      return;
    }
    if (!bH.get_rep()) {
      error("Empty B matrix for Add");
      return;
    }
    if (aH->ncols() != bH->ncols() || aH->nrows() != bH->nrows())
    {
      error("Addition requires A and B must be the same size.");
      return;
    }
    omat_->send(aH + bH);
    return;
  } else if (op == "Mult") {
    if (!aH.get_rep()) {
      error("Empty A matrix for Mult");
      return;
    }
    if (!bH.get_rep()) {
      error("Empty B matrix for Mult");
      return;
    }
    if (aH->ncols() != bH->nrows())
    {
      error("Matrix multiply requires the number of columns in A to be the same as the number of rows in B.");
      error("A contains " + to_string(aH->ncols()) +
            " columns, B contains " + to_string(bH->nrows()) + " rows.");
      return;
    }
    omat_->send(aH * bH);
    return;
  } else if (op == "Function") {
    if (aH->nrows()*aH->ncols() != bH->nrows()*bH->ncols()) {
      error("Function only works if input matrices have the same number of elements.");
      return;
    }
    
    MatrixHandle aHtmp = aH;
    MatrixHandle bHtmp = bH;
    if (aH->is_sparse()) { aHtmp = aH->dense(); }
    if (bH->is_sparse()) { bHtmp = bH->dense(); }

    // Remove trailing white-space from the function string.
    string func = function_.get();
    while (func.size() && isspace(func[func.size()-1]))
    {
      func.resize(func.size()-1);
    }

    // Compile the function.
    int hoffset = 0;
    Handle<LinAlgBinaryAlgo> algo;
    for( ;; )
    {
      CompileInfoHandle ci =
	LinAlgBinaryAlgo::get_compile_info(func, hoffset);
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
    MatrixHandle m;
    if (aH->ncols() == 1)
    {
      m = scinew ColumnMatrix(aH->nrows());
    }
    else
    {
      m = scinew DenseMatrix(aH->nrows(), aH->ncols());
    }
    double *a = aHtmp->get_data_pointer();
    double *b = bHtmp->get_data_pointer();
    double *x = m->get_data_pointer();
    const int n = m->nrows() * m->ncols();
    for (int i = 0; i < n; i++)
    {
      x[i] = algo->user_function(a[i], b[i]);
    }
    omat_->send(m);

  } else if (op == "SelectColumns") {
    if (!aH.get_rep() || !bH.get_rep()) {
      error("Can't have an empty input matrix for SelectColumns.");
      return;
    }
    ColumnMatrix *bc = dynamic_cast<ColumnMatrix *>(bH.get_rep());
    if (!bc) {
      error("Second input to SelectColumns must be a ColumnMatrix.");
      return;
    }
    DenseMatrix *cd = scinew DenseMatrix(aH->nrows(), bc->nrows());
    for (int i=0; i<cd->ncols(); i++) {
      int idx = (int)(*bc)[i];
      if (idx == -1) continue;
      if (idx > aH->ncols()) {
	error("Tried to select column (" + to_string(idx) +
	      ") that was out of range (" + to_string(aH->ncols()) + ").");
	return;
      }
      for (int j=0; j<aH->nrows(); j++) {
	(*cd)[j][i]=aH->get(j,idx);
      }
    }
    if (dynamic_cast<DenseMatrix *>(aH.get_rep()))
      omat_->send(MatrixHandle(cd));
    else if (dynamic_cast<ColumnMatrix *>(aH.get_rep())) {
      omat_->send(MatrixHandle(cd->column()));
      delete cd;
    } else {
      omat_->send(MatrixHandle(cd->sparse()));
      delete cd;
    }
    return;
  } else if (op == "SelectRows") {
    if (!aH.get_rep() || !bH.get_rep()) {
      error("Can't have an empty input matrix for SelectRows.");
      return;
    }
    ColumnMatrix *bc = dynamic_cast<ColumnMatrix *>(bH.get_rep());
    if (!bc) {
      error("Second input must be a ColumnMatrix for SelectRows.");
      return;
    }
    DenseMatrix *cd = scinew DenseMatrix(bc->nrows(), aH->ncols());
    for (int i=0; i<cd->nrows(); i++) {
      int idx = (int)(*bc)[i];
      if (idx == -1) continue;
      if (idx > aH->nrows()) {
	error("Tried to select a row (" + to_string(idx) +
	      ") that was out of range (" + to_string(aH->nrows()) +").");
	return;
      }
      for (int j=0; j<aH->ncols(); j++) {
	(*cd)[i][j]=aH->get(idx,j);
      }
    }
    if (dynamic_cast<DenseMatrix *>(aH.get_rep()))
      omat_->send(MatrixHandle(cd));
    else if (dynamic_cast<ColumnMatrix *>(aH.get_rep())) {
      omat_->send(MatrixHandle(cd->column()));
      delete cd;
    } else {
      omat_->send(MatrixHandle(cd->sparse()));
      delete cd;
    }
    return;
  } else if (op == "NormalizeAtoB") {
    if (!aH.get_rep() || !bH.get_rep()) {
      error("Can't have an empty input matrix for NormalizeAtoB.");
      return;
    }

    if (aH->is_sparse() || bH->is_sparse())
    {
      error("NormalizeAtoB does not currently support SparseRowMatrices.");
      return;
    }
    
    if (aH->get_data_size() == 0 || bH->get_data_size() == 0)
    {
      error("Cannot NormalizeAtoB with empty matrices.");
      return;
    }
    
    double amin, amax, bmin, bmax;
    double *a = aH->get_data_pointer();
    double *b = bH->get_data_pointer();
    const int na = aH->get_data_size();
    const int nb = bH->get_data_size();
    amin = amax = a[0];
    bmin = bmax = b[0];
    int i;
    for (i=1; i<na; i++) {
      if (a[i]<amin) amin=a[i];
      else if (a[i]>amax) amax=a[i];
    }
    for (i=1; i<nb; i++) {
      if (b[i]<bmin) bmin=b[i];
      else if (b[i]>bmax) bmax=b[i];
    }

    MatrixHandle anewH = aH->clone();
    double *anew = anewH->get_data_pointer();
    const double scale = (bmax - bmin)/(amax - amin);
    for (i=0; i<na; i++)
      anew[i] = (a[i]-amin)*scale+bmin;
    omat_->send(anewH);
  } else {
    warning("Don't know operation "+op);
    return;
  }
}


CompileInfoHandle
LinAlgBinaryAlgo::get_compile_info(const string &function,
				  int hashoffset)

{
  unsigned int hashval = Hash(function, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("LinAlgBinaryInstance" + to_string(hashval));
  static const string base_class_name("LinAlgBinaryAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + ".",
		       base_class_name,
		       template_name + ";//",
		       "");

  // Code for the function.
  string class_declaration =
    string("using namespace SCIRun;\n\n") + 
    "class " + template_name + " : public LinAlgBinaryAlgo\n" +
    "{\n" +
    "  virtual double user_function(double x, double y)\n" +
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
