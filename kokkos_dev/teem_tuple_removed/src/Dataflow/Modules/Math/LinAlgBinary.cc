
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
#include <Dataflow/Modules/Math/LinAlgBinary.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <iostream>
#include <sstream>
#include <sci_hash_map.h>

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

  if (!imatA_) {
    error("Unable to initialize iport 'A'.");
    return;
  }
  if (!imatB_) {
    error("Unable to initialize iport 'B'.");
    return;
  }
  if (!omat_) {
    error("Unable to initialize oport 'Output'.");
    return;
  }
  
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
      return;
    }
    omat_->send(aH * bH);
    return;
  } else if (op == "Function") {
    if (aH->nrows()*aH->ncols() != bH->nrows()*bH->ncols()) {
      error("Function only works if input matrices have the same number of elements.");
      return;
    }

    // Remove trailing white-space from the function string.
    string func = function_.get();
    while (func.size() && isspace(func[func.size()-1]))
    {
      func.resize(func.size()-1);
    }

    // Compile the function.
    int hoffset = 0;
    Handle<LinAlgBinaryAlgo> algo;
    while (1)
    {
      CompileInfoHandle ci =
	LinAlgBinaryAlgo::get_compile_info(func, hoffset);
      if (!DynamicCompilation::compile(ci, algo, false, this))
      {
	DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
	error("Your function would not compile.");
	return;
      }
      if (algo->identify() == func)
      {
	break;
      }
      hoffset++;
    }

    // Get the data from the matrix, iterate over it calling the function.
    MatrixHandle m = aH->clone();
    double *a = &((*(aH.get_rep()))[0][0]);
    double *b = &((*(bH.get_rep()))[0][0]);
    double *x = &((*(m.get_rep()))[0][0]);
    const int n = m->nrows() * m->ncols();
    for (int i = 0; i < n; i++)
    {
      x[i] = algo->user_function(a[i], b[i]);
    }
    omat_->send(m);

  } else if (op == "SelectColumns") {
    if (!aH.get_rep() || !bH.get_rep()) {
      cerr << "LinAlgBinary:SelectColumns Error - can't have an empty input matrix for this operation.\n";
      return;
    }
    ColumnMatrix *bc = dynamic_cast<ColumnMatrix *>(bH.get_rep());
    if (!bc) {
      cerr << "LinAlgBinary:SelectColumns Error - second input must be a ColumnMatrix!\n";
      return;
    }
    DenseMatrix *cd = scinew DenseMatrix(aH->nrows(), bc->nrows());
    for (int i=0; i<cd->ncols(); i++) {
      int idx = (int)(*bc)[i];
      if (idx == -1) continue;
      if (idx > aH->ncols()) {
	cerr << "LinAlgBinary:SelectColumns Error - tried to select a column ("<<idx<<") that was out of range ("<<aH->ncols()<<")\n";
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
      cerr << "LinAlgBinary:SelectRows Error - can't have an empty input matrix for this operation.\n";
      return;
    }
    ColumnMatrix *bc = dynamic_cast<ColumnMatrix *>(bH.get_rep());
    if (!bc) {
      cerr << "LinAlgBinary:SelectRows Error - second input must be a ColumnMatrix!\n";
      return;
    }
    DenseMatrix *cd = scinew DenseMatrix(bc->nrows(), aH->ncols());
    for (int i=0; i<cd->nrows(); i++) {
      int idx = (int)(*bc)[i];
      if (idx == -1) continue;
      if (idx > aH->nrows()) {
	cerr << "LinAlgBinary:SelectRows Error - tried to select a row ("<<idx<<") that was out of range ("<<aH->nrows()<<")\n";
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
      cerr << "LinAlgBinary:NormalizeAtoB Error - can't have an empty input matrix for this operation.\n";
      return;
    }
    double amin, amax, bmin, bmax;
    MatrixHandle anewH = aH->clone();
    double *a = &((*(aH.get_rep()))[0][0]);
    double *anew = &((*(anewH.get_rep()))[0][0]);
    double *b = &((*(bH.get_rep()))[0][0]);
    int na = aH->nrows()*aH->ncols();
    int nb = bH->nrows()*bH->ncols();
    amin=amax=a[0];
    bmin=bmax=b[0];
    int i;
    for (i=1; i<na; i++) {
      if (a[i]<amin) amin=a[i];
      else if (a[i]>amax) amax=a[i];
    }
    for (i=1; i<nb; i++) {
      if (b[i]<bmin) bmin=b[i];
      else if (b[i]>bmax) bmax=b[i];
    }
    double da=amax-amin;
    double db=bmax-bmin;
    double scale = db/da;
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
  hash<const char *> H;
  unsigned int hashval = H(function.c_str()) + hashoffset;

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
    string("\"\n\nusing namespace SCIRun;\n\n") + 
    "class " + template_name + " : public LinAlgBinaryAlgo\n" +
    "{\n" +
    "  virtual double user_function(double x, double y)\n" +
    "  {\n" +
    "    return (" + function + ");\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + string_Cify(function) + "\"); }\n" +
    "};\n//";

  // Add in the include path to compile this obj
  rval->add_include(include_path + class_declaration);
  return rval;
}


} // End namespace SCIRun
