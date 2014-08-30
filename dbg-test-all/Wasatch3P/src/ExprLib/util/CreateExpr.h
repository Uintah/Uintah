#ifndef CreateExpr_h
#define CreateExpr_h

#include <expression/Tag.h>

#include <sstream>
#include <vector>
#include <string>

//====================================================================

class Indent
{
  size_t n_;
  const size_t tab_;
public:
  Indent( const size_t n=0,
          const size_t tab=2 );
  Indent& operator++(){ n_+=tab_; return *this; }
  Indent& operator--(){ n_-=tab_; return *this; }

  Indent& operator+=(const size_t n){ n_+=n; return *this; }
  Indent& operator-=(const size_t n){ n_-=n; return *this; }
  Indent& operator =(const size_t n){ n_=n; return *this; }

  std::string str() const;
};

std::ostream& operator << ( std::ostream&, const Indent& );

//====================================================================

struct FieldInfo{
  std::string fieldname, fieldtype;
  FieldInfo( const std::string name, const std::string n ) : fieldname(name), fieldtype(n) {}
  std::string get_tag_name() const;
  std::string get_var_name() const;
};

typedef std::vector<FieldInfo> FieldDeps;

class Info
{
public:

  enum Option{
    FIELD_TYPE_NAME,		///< Name for Field type that the expression computes
    EXPR_NAME,			///< Name of the expression
    FILE_NAME,			///< Name of the file
    EXTRA_TEMPLATE_PARAMS,	///< Template parameters
  };

  Info();
  Info( const std::string fieldTypeName,
        const std::string exprName,
        const std::string fileName );

  void clear();

  bool is_field_template_param() const{ return isFieldTParam_; }

  void set( const Option, const std::string val );

  void set_dep_field( const std::string name,
                      const std::string fieldTypeName );

  std::string get( const Option ) const;

  const FieldDeps& get_dep_fields() const;

  void finalize();

  const std::vector<std::string>& get_template_params() const;

  size_t get_n_template_params() const;

  std::ostream& print(std::ostream& os) const;

private:
  std::string exprName_,
    fileName_,
    fieldTypeName_;

  std::vector<std::string> templateParamNames_;

  FieldDeps depFields_;

  bool isFieldTParam_;
  bool isFinalized_;
};

std::ostream& operator<<(std::ostream&, const Info&);

//====================================================================

enum HeaderOption{
  CLASS_DECLARE,
  CLASS_IMPLEMENT,
  METHOD_IMPLEMENT
};

//====================================================================

std::string
template_header( Indent indent,
                 const HeaderOption op,
                 const Info& info );

std::string
method_separator( const std::string c="-" );

//====================================================================

class MethodWriter
{
protected:
  const HeaderOption op_;
  const Info& info_;
  Indent indent_;
  std::ostringstream outbuf_;
public:
  MethodWriter( Indent& indent,
                const HeaderOption op,
                const Info& info,
                const std::string returnType,
                const std::string methodDeclare,
                const std::string methodImplement="" );
  virtual ~MethodWriter(){};
  virtual std::ostream& put(std::ostream&) const;
};

std::ostream&
operator << ( std::ostream&, const MethodWriter& );

//====================================================================

class ConstructorWriter : public MethodWriter
{
  static std::string method_text( Indent indent, const HeaderOption, const Info& );
public:
  ConstructorWriter( Indent indent, const HeaderOption op, const Info& );
};

struct DestructorWriter : public MethodWriter{
  DestructorWriter( Indent indent,const HeaderOption op, const Info& );
};

class AdvertDepWriter : public MethodWriter
{
  static std::string method_text( Indent indent, const HeaderOption op, const Info& );
  static std::string implement_text( Indent indent, const Info& );
public:
  AdvertDepWriter( Indent indent, const HeaderOption op, const Info& );
};

class BindFieldWriter : public MethodWriter
{
  static std::string method_text( Indent indent, const HeaderOption op, const Info& );
  static std::string implement_text( Indent indent, const Info& );
public:
  BindFieldWriter( Indent indent, const HeaderOption op,const Info& );
};

class BindOpWriter : public MethodWriter
{
public:
  BindOpWriter( Indent indent, const HeaderOption op,const Info& );
  static std::string implement_text( Indent indent );
};

class EvalWriter : public MethodWriter
{
public:
  EvalWriter( Indent indent, const HeaderOption op,const Info& );
  static std::string implement_text( Indent indent, const Info& );
};

//====================================================================

class CreateExpr
{
  const Info info_;
  std::ostringstream out_;
  Indent indent_;

  void write_preamble();
  void write_var_declarations();
  void write_builder( const HeaderOption );

  void write_methods( const HeaderOption );

public:
  CreateExpr( const Info& info );
  ~CreateExpr();
  std::ostringstream& get_stream(){ return out_; }
};

//====================================================================

#endif // CreateExpr_h
