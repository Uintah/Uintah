#include <iostream>
#include <cassert>
#include <stdexcept>

#include "CreateExpr.h"

using namespace std;

Indent::Indent( const size_t n,
                const size_t tab )
  : n_( n ),
    tab_( tab )
{}

string
Indent::str() const
{
  return string(n_,' ');
}

ostream&
operator<<( ostream& os, const Indent& indent )
{
  os << indent.str();
  return os;
}

//====================================================================

std::string
FieldInfo::get_tag_name() const
{
  return fieldname + "Tag";
}

std::string
FieldInfo::get_var_name() const
{
  return fieldname + "_";
}

//====================================================================

Info::Info()
{
  isFinalized_ = false;
  fieldTypeName_ = "FieldT";  isFieldTParam_ = false;
}

Info::Info( const std::string fieldTypeN,
            const std::string exprN,
            const std::string fileN )
  : exprName_( exprN ),
    fileName_( fileN ),
    fieldTypeName_( fieldTypeN )
{
  isFinalized_   = false;
  isFieldTParam_ = false;
}

void
Info::clear()
{
  exprName_ = "";
  fileName_ = "";
  fieldTypeName_ = "";
  templateParamNames_.clear();
  depFields_.clear();
  isFinalized_ = false;
}

void
Info::set( const Option op, const string val )
{
  assert( !isFinalized_ );
  if( isFinalized_ ) return;
  switch(op) {
  case FILE_NAME :
    fileName_ = val;
    break;
  case EXPR_NAME :
    exprName_ = val;
    break;
  case FIELD_TYPE_NAME :
    fieldTypeName_      = val;
    break;
  case EXTRA_TEMPLATE_PARAMS:
    if( val==fieldTypeName_ ) isFieldTParam_ = true;
    templateParamNames_.push_back(val);
    break;
  default:
    throw std::runtime_error("invalid case in switch statement");
  }
}

void
Info::set_dep_field( const std::string name,
                     const std::string fieldTypeName )
{
  depFields_.push_back( FieldInfo(name,fieldTypeName) );
}

const std::vector<FieldInfo>&
Info::get_dep_fields() const
{
  return depFields_;
}

string
Info::get( const Option op ) const
{
  string val;
  switch(op) {
  case FILE_NAME       : val = fileName_;      break;
  case EXPR_NAME       : val = exprName_;      break;
  case FIELD_TYPE_NAME : val = fieldTypeName_; break;
  default: assert(0); break;
  }
  return val;
}

const vector<string>&
Info::get_template_params() const
{
  return templateParamNames_;
}

size_t
Info::get_n_template_params() const
{
  return templateParamNames_.size();
}

void
Info::finalize()
{
  isFinalized_ = true;
  if( fieldTypeName_.empty() )  fieldTypeName_ = "FieldT";
}

ostream&
Info::print( ostream& os ) const
{
  os << "File name       : " << fileName_ << endl
     << "Expression name : " << exprName_ << endl
     << "Field Type name : " << fieldTypeName_ << endl;
  if( !templateParamNames_.empty() ){
    os << "Template params : " << endl
       << "                  < ";
    const int n = templateParamNames_.size();
    for( int i=0; i<n; ++i ){
      os << templateParamNames_[i];
      if( i!=n-1 ) os << ", ";
    }
    os << " >" << endl;
  }
  if( !depFields_.empty() ){
    os << "Dependencies    : " << endl;
    for( FieldDeps::const_iterator idep = depFields_.begin(); idep!=depFields_.end(); ++idep ){
      os << "                  Field=" << idep->fieldname
         << ",  Type=" << idep->fieldtype << endl;
    }
    os << endl;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Info& info )
{
  info.print(os);
  return os;
}

//====================================================================

string
template_header( Indent indent,
                 const HeaderOption op,
                 const Info& info )
{
  ostringstream out;
  const vector<string>& tpars = info.get_template_params();

  switch(op) {
  case CLASS_DECLARE  :
    out << "/**" << endl
        << " *  \\class " << info.get(Info::EXPR_NAME) << endl
        << " */" << endl;
  case CLASS_IMPLEMENT:
    if( !info.get_template_params().empty() )
      out << "template<";
    break;
  case METHOD_IMPLEMENT :
    if( !info.get_template_params().empty() )
      out << "<";
    break;
  }

  vector<string>::const_iterator ip=tpars.begin();
  if( !tpars.empty() ){
    if( op==CLASS_DECLARE || op==CLASS_IMPLEMENT )
      out << indent << " typename ";
    out << *ip;
    ++ip;
    if( op==CLASS_DECLARE || op==CLASS_IMPLEMENT )
      indent += 9;
  }
  for( ; ip!=tpars.end(); ++ip ){
    out << ",";
    switch(op){
    case CLASS_DECLARE:
      out << endl << indent;
    case CLASS_IMPLEMENT:
      out << " typename ";
      break;
    case METHOD_IMPLEMENT:
      break;
    }
    out << *ip;
  }
  if( op==CLASS_DECLARE || op==CLASS_IMPLEMENT ){
    indent-=9;
    if( !info.get_template_params().empty() )
      out << " >" << endl;
  }
  else{
    if( !info.get_template_params().empty() )
      out << ">";
  }
  if( op==METHOD_IMPLEMENT )
    out << "::" << endl;

  switch(op){
  case CLASS_DECLARE :{
    const string& fieldTName = info.get(Info::FIELD_TYPE_NAME);
    out << "class " << info.get(Info::EXPR_NAME) << endl
        << " : public Expr::Expression<"
        << fieldTName
        << ">" << endl  // jcs what if we have extra template parameters?
        << "{" << endl;
    break;
  }
  case CLASS_IMPLEMENT: case METHOD_IMPLEMENT:
    break;
  }
  return out.str();
}

//====================================================================

string
method_separator( const std::string c )
{
  string line("//");
  line.append(68,'-');
  return line+"\n";
}

//====================================================================

MethodWriter::MethodWriter( Indent& indent,
                            const HeaderOption op,
                            const Info& info,
                            const string returnType,
                            const string methodDeclare,
                            const string methodImplement )
  : op_( op ),
    info_( info ),
    indent_( indent )
{
  switch(op){

  case CLASS_DECLARE:
    if( returnType.empty() )
      outbuf_ << indent << methodDeclare << ";" << endl;
    else
      outbuf_ << indent << returnType << " " << methodDeclare << ";" << endl;
    break;

  case CLASS_IMPLEMENT:
  case METHOD_IMPLEMENT:
    outbuf_ << template_header( indent, CLASS_IMPLEMENT, info );
    if( ! returnType.empty() )
      outbuf_ << returnType << endl;
    outbuf_ << info.get(Info::EXPR_NAME)
            << template_header( indent, METHOD_IMPLEMENT, info )
            << methodDeclare << endl
            << "{";
    if( !methodImplement.empty() )
      outbuf_ << methodImplement << endl;
    outbuf_ << "}" << endl
            << endl
            << method_separator()
            << endl;
    break;
  } // switch(op)
}

ostream&
MethodWriter::put(ostream& os) const
{
  return os << outbuf_.str();
}

ostream&
operator << ( ostream& os, const MethodWriter& mw )
{
  return mw.put(os);
}

//====================================================================

ConstructorWriter::
ConstructorWriter( Indent indent,
                   const HeaderOption op,
                   const Info& info )
  : MethodWriter( indent, op, info, "", method_text(indent,op,info) )
{}

std::string
ConstructorWriter::
method_text( Indent indent, const HeaderOption op, const Info& info )
{
  ostringstream out;
  const string cnam = info.get(Info::EXPR_NAME) + "(";
  out << indent << cnam;
  const FieldDeps& fd = info.get_dep_fields();
  if( fd.empty() ){
    out << " /* class-specific arguments (typically Expr::Tag objects) */";
  }
  for( size_t i=0; i<fd.size(); ++i ){
    if( i>0 ){
      out << "," << endl
          << indent << "const Expr::Tag& " << fd[i].get_tag_name();
    }
    else{
      indent += cnam.length() + 2;
      out << " const Expr::Tag& " << fd[i].get_tag_name();
    }
  }
  out << " )";
  if( op!=CLASS_DECLARE ){
    indent = 2;
    out << endl
        << indent
        << ": Expr::Expression<"
        << info.get(Info::FIELD_TYPE_NAME)
        << ">()";
    ++indent;
    for( size_t i=0; i<fd.size(); ++i ){
      out << "," << endl << indent << fd[i].get_tag_name() << "_( " << fd[i].get_tag_name() << " )";
    }
    --indent;
  }
  return out.str();
}

//====================================================================

DestructorWriter::
DestructorWriter( Indent indent,
                  const HeaderOption op,
                  const Info& info )
  : MethodWriter( indent, op, info, "",
                  "~"+info.get(Info::EXPR_NAME)+"()" )
{}

//====================================================================

std::string
AdvertDepWriter::
method_text( Indent indent, const HeaderOption op, const Info& info )
{
  ostringstream out;
  const string tmp = "advertise_dependents( ";
  if( op!=CLASS_DECLARE ) out << indent;
  out << tmp << "Expr::ExprDeps& exprDeps )";
  return out.str();
}

std::string
AdvertDepWriter::
implement_text( Indent indent, const Info& info )
{
  ++indent;
  ostringstream out;
  if( info.get_dep_fields().empty() ){
    out << endl
        << indent
        << "/* add dependencies as follows (TAG represents the Expr::Tag for the depenency): */" << endl
        << indent
        << "// exprDeps.requires_expression( TAG );";
  }
  else{
    const FieldDeps& fd = info.get_dep_fields();
    for( FieldDeps::const_iterator idep=fd.begin(); idep!=fd.end(); ++idep ){
      out << endl << indent
          << "exprDeps.requires_expression( " << idep->get_tag_name() << "_ );";
    }
  }
  return out.str();
}

AdvertDepWriter::
AdvertDepWriter( Indent indent,
                 const HeaderOption op,
                 const Info& info )
  : MethodWriter( indent, op, info, "void",
                  method_text( indent, op, info ),
                  implement_text(indent,info) )
{}

//====================================================================

std::string
BindFieldWriter::
method_text( Indent indent, const HeaderOption op, const Info& info )
{
  ostringstream out;
  out << "bind_fields( const Expr::FieldManagerList& fml )";
  return out.str();
}

std::string
BindFieldWriter::
implement_text( Indent indent, const Info& info )
{
  const bool hasTemplate = info.get_n_template_params() > 0;
  const string templateAddendum = hasTemplate ? "template " : "";
  ++indent;
  ostringstream out;
  out << endl
      << indent << "/* add additional code here to bind any fields required by this expression */" << endl << endl
      << indent << "/* if you have many fields of the same type to bind, first get the FieldManager:" << endl
      << indent << "     const ";
  if( hasTemplate ) out << "typename ";
  out << "Expr::FieldMgrSelector<"
      << info.get(Info::FIELD_TYPE_NAME)
      << ">::type& fm = fml." << templateAddendum << "field_manager<"
      << info.get(Info::FIELD_TYPE_NAME) << ">();" << endl
      << indent << "   and then get fields:" << endl
      << indent << "     myField1_ = &fm.field_ref(myTag1_);" << endl
      << indent << "     myField2_ = &fm.field_ref(myTag2_);" << endl
      << indent << "     myField3_ = &fm.field_ref(myTag3_);" << endl
      << indent << "   this can yield slightly better performance."
      << indent << "*/" << endl << endl;

  // write out requested dependent variables
  const FieldDeps& fd = info.get_dep_fields();
  for( FieldDeps::const_iterator idep=fd.begin(); idep!=fd.end(); ++idep ){
    out << endl << indent << idep->get_var_name() << " = &fml.";
    if( hasTemplate > 0 ) out << "template ";
    out << "field_ref< " << idep->fieldtype << " >( "
        << idep->get_tag_name() << "_ );";
  }
  return out.str();
}

BindFieldWriter::
BindFieldWriter( Indent indent,
                 const HeaderOption op,
                 const Info& info )
  : MethodWriter( indent, op, info, "void",
                  method_text( indent, op, info ),
                  implement_text(indent,info) )
{}

//====================================================================

BindOpWriter::
BindOpWriter( Indent indent,
              const HeaderOption op,
              const Info& info )
  : MethodWriter( indent, op, info, "void",
                  "bind_operators( const SpatialOps::OperatorDatabase& opDB )",
                  implement_text(indent) )
{}

std::string
BindOpWriter::
implement_text( Indent indent )
{
  ++indent;
  ostringstream out;
  out << endl
      << indent << "// bind operators as follows:" << endl
      << indent << "// op_ = opDB.retrieve_operator<OpT>();";
  return out.str();
}

//====================================================================

EvalWriter::
EvalWriter( Indent indent,
            const HeaderOption op,
            const Info& info )
  : MethodWriter( indent,
                  op,
                  info,
                  "void",
                  "evaluate()",
                  implement_text(indent,info) )
{}

std::string
EvalWriter::implement_text( Indent indent, const Info& info )
{
  ++indent;
  ostringstream out;
  out << endl
      << indent << info.get(Info::FIELD_TYPE_NAME) << "& result = this->value();" << endl
      << endl
      << indent << "/* evaluation code goes here - be sure to assign the appropriate value to 'result' */";
  return out.str();
}

//====================================================================

CreateExpr::CreateExpr( const Info& info )
  : info_( info ),
    //    out_( (info.get(Info::FILE_NAME)+".h").c_str(), ios_base::out),
    indent_( 0, 2 )
{
  write_preamble();

  ++indent_;

  write_var_declarations();
  ConstructorWriter cw( indent_, CLASS_DECLARE, info );  out_ << cw;

  --indent_;  out_ << "public:" << endl;  ++indent_;

  write_builder( CLASS_DECLARE );
  write_methods( CLASS_DECLARE );

  out_ << "};" << endl;

  indent_ = 0;
  out_ << endl << endl << endl
        << "// " << string(67,'#') << endl << "//" << endl
        << "// " << string(25,' ') << "Implementation" << endl
        << "//" << endl << "// " << string(67,'#') << endl
        << endl << endl << endl;

  ConstructorWriter cw2( indent_, METHOD_IMPLEMENT, info );  out_ << cw2;
  write_methods( METHOD_IMPLEMENT );
  write_builder( METHOD_IMPLEMENT );

  out_ << endl;

  out_ << endl
        << "#endif // " << info_.get(Info::FILE_NAME) << "_Expr_h" << endl;
}

CreateExpr::~CreateExpr()
{}

void
CreateExpr::write_preamble()
{
  const string fnam = info_.get(Info::FILE_NAME);
  out_ << "#ifndef " << fnam << "_Expr_h" << endl
        << "#define " << fnam << "_Expr_h" << endl
        << endl
        << "#include <expression/Expression.h>" << endl
        << endl;
  if( !info_.is_field_template_param() ){
    out_ << "// DEFINE THE TYPE OF FIELD FOR THIS EXPRESSION HERE" << endl
          << "typedef /* insert field type here */ "
          << info_.get(Info::FIELD_TYPE_NAME) << ";" << endl << endl;
  }
  out_ << template_header( indent_, CLASS_DECLARE, info_ );
}

void
CreateExpr::write_var_declarations()
{
  const FieldDeps& fd = info_.get_dep_fields();
  const size_t n = fd.size();
  if( n==0 ){
    out_ << indent_ << "/* declare tags that need to be saved here:" << endl
         << indent_ << "Expr::Tag myVar;" << endl << indent_ << "*/" << endl;
  }
  else{
    out_ << indent_ << "const Expr::Tag ";
    for( size_t i=0; i<n; ++i ){
      if( i>0 ) out_ << ", ";
      out_ << fd[i].get_tag_name() << "_";
    }
    out_ << ";" << endl;
  }

  if( fd.empty() ){
    out_ << endl << indent_ << "/* declare private variables here */" << endl;
  }
  for( FieldDeps::const_iterator idep=fd.begin(); idep!=fd.end(); ++idep ){
    out_ << indent_ << "const " << idep->fieldtype << "* " << idep->get_var_name() << ";" << endl;
  }

  out_ << endl << indent_ << "/* declare operators associated with this expression here */"
        << endl << endl;
}

void
CreateExpr::write_builder( const HeaderOption op )
{
  const FieldDeps& fd = info_.get_dep_fields();

  switch(op){
  case CLASS_DECLARE:
    {
      out_ << indent_ << "class Builder : public Expr::ExpressionBuilder" << endl
            << indent_ << "{" << endl
            << indent_ << "public:" << endl;
      ++indent_;
      out_ << indent_ << "/**" << endl
           << indent_ << " *  @brief Build a " << info_.get(Info::EXPR_NAME) << " expression" << endl
           << indent_ << " *  @param resultTag the tag for the value that this expression computes" << endl
           << indent_ << " */" << endl
           << indent_ << "Builder( const Expr::Tag& resultTag";
      if( fd.size()==0 ){
        out_ << " /* add additional arguments here */";
      }
      else{
        const FieldDeps& fd = info_.get_dep_fields();
        indent_ += 9;
        for( size_t i=0; i<fd.size(); ++i ){
          out_ << "," << endl << indent_ << "const Expr::Tag& " << fd[i].get_tag_name();
        }
        indent_ -= 9;
      }
      out_ << " );" << endl
            << endl
            << indent_ << "Expr::ExpressionBase* build() const;" << endl
            << endl;
      --indent_;
      out_ << indent_ << "private:" << endl;
      ++indent_;
      const FieldDeps& fd = info_.get_dep_fields();
      const size_t n=fd.size();
      if( n==0 ){ out_ << indent_ << " /* add additional arguments here */" << endl; }
      else{
        out_ << indent_ << "const Expr::Tag ";
        for( size_t i=0; i<n; ++i ){
          if( i>0 ) out_ << ", ";
          out_ << fd[i].get_tag_name() << "_";
        }
        out_ << ";" << endl;
      }
      --indent_;
      out_ << indent_ << "};" << endl << endl;
      break;
    }

  case CLASS_IMPLEMENT:
  case METHOD_IMPLEMENT:
    {
      out_ << template_header( indent_, CLASS_IMPLEMENT,  info_ )
            << info_.get(Info::EXPR_NAME)
            << template_header( indent_, METHOD_IMPLEMENT, info_ )
            << "Builder::Builder( const Expr::Tag& resultTag";
      if( fd.size()==0 ){
        out_ << " /* add arguments here */";
      }
      else{
        const string cnam = info_.get(Info::EXPR_NAME);
        const int nindent = 4 + 2*string("Builder").length();
        const size_t n = fd.size();
        for( size_t i=0; i<n; ++i ){
          if( i==0 ) indent_ += nindent;
          out_ << "," << endl << indent_ << "const Expr::Tag& " << fd[i].get_tag_name();
        }
        if( n>0 ) indent_ -= nindent;
      }
      ++indent_;
      out_ << " )" << endl << indent_ << ": ExpressionBuilder( resultTag )";
      if( !fd.empty() ){
        ++indent_;
        const size_t n = fd.size();
        for( size_t i=0; i<n; ++i ){
          out_ << "," << endl << indent_ << fd[i].get_tag_name() << "_( " << fd[i].get_tag_name() << " )";
        }
        --indent_;
      }
      --indent_;
      out_ << endl;
      out_ << indent_ << "{}" << endl << endl
            << method_separator() << endl
            << template_header( indent_, CLASS_IMPLEMENT,  info_ )
            << "Expr::ExpressionBase*" << endl
            << info_.get(Info::EXPR_NAME)
            << template_header( indent_, METHOD_IMPLEMENT, info_ )
            << "Builder::build() const" << endl
            << "{" << endl
            << "  return new " << info_.get(Info::EXPR_NAME);
      if( info_.get_n_template_params()>0 ){
        out_ << "<";
        const vector<string>& params = info_.get_template_params();
        vector<string>::const_iterator ip=params.begin();
        if( !params.empty() ){
          out_ << *ip;
          ++ip;
        }
        for( ; ip!=params.end(); ++ip ){
          out_ << "," << *ip;
        }
        out_ << ">";
      }
      out_ << "( ";
      const size_t n = fd.size();
      if( n==0 )  out_ << "/* insert additional arguments here */";
      for( size_t i=0; i<n; ++i ){
        if( i>0 ) out_ << ",";
        out_ << fd[i].get_tag_name() << "_";
      }
      out_ << " );" << endl
            << "}" << endl;
    }
  }
}

void
CreateExpr::write_methods( const HeaderOption op )
{
  DestructorWriter dw( indent_, op, info_ );  out_ << dw;
  AdvertDepWriter adw( indent_, op, info_ );  out_ << adw;
  BindFieldWriter bfw( indent_, op, info_ );  out_ << bfw;
  BindOpWriter    bow( indent_, op, info_ );  out_ << bow;
  EvalWriter      evw( indent_, op, info_ );  out_ << evw;
}

//====================================================================
