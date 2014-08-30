#include <expression/Expression.h>

//====================================================================

class A : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( bt_ );
    exprDeps.requires_expression( ct_ );
    exprDeps.requires_expression( dt_ );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    const Expr::FieldMgrSelector<SpatialOps::SingleValueField>::type& fmgr = fml.field_manager<SpatialOps::SingleValueField>();
    b_ = &fmgr.field_ref(bt_);
    c_ = &fmgr.field_ref(ct_);
    d_ = &fmgr.field_ref(dt_);
  }

  void evaluate()
  {
    using namespace SpatialOps;
    this->value() <<= 1.1 + *b_ + *c_ + *d_;
  }

  ~A(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const
    {
      return new A( bt_, ct_, dt_ );
    }
    Builder( const Expr::Tag& aTag,
             const Expr::Tag& bTag,
             const Expr::Tag& cTag,
             const Expr::Tag& dTag )
      : ExpressionBuilder(aTag),
        bt_(bTag), ct_(cTag), dt_(dTag)
    {}
    ~Builder(){}
  private:
    const Expr::Tag bt_, ct_, dt_;
  };

protected:
  A( const Expr::Tag& bTag,
     const Expr::Tag& cTag,
     const Expr::Tag& dTag )
    : Expr::Expression<SpatialOps::SingleValueField>(),
      bt_(bTag), ct_(cTag), dt_(dTag)
  {
    this->set_gpu_runnable(true);
  }
  const Expr::Tag bt_, ct_, dt_;
  const SpatialOps::SingleValueField *b_, *c_, *d_;
};

//====================================================================

class B : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( eTag_ );
    exprDeps.requires_expression( gTag_ );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    const Expr::FieldMgrSelector<SpatialOps::SingleValueField>::type& fmgr = fml.field_manager<SpatialOps::SingleValueField>();
    e_ = &fmgr.field_ref( eTag_ );
    g_ = &fmgr.field_ref( gTag_ );
  }

  void evaluate(){
    using namespace SpatialOps;
    this->value() <<= 2.2 + *e_ + *g_;
  }

  ~B(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const{ return new B(eTag_,gTag_); }
    Builder( const Expr::Tag& bTag,
             const Expr::Tag& eTag,
             const Expr::Tag& gTag )
    : ExpressionBuilder(bTag),
      eTag_(eTag), gTag_(gTag)
    {}
    ~Builder(){}
  private:
    const Expr::Tag eTag_, gTag_;
  };

protected:
  B( const Expr::Tag& etag,
     const Expr::Tag& gtag )
    : Expr::Expression<SpatialOps::SingleValueField>(),
      eTag_(etag), gTag_(gtag)
  {
    this->set_gpu_runnable(true);
  }
  const Expr::Tag eTag_, gTag_;
  const SpatialOps::SingleValueField *e_, *g_;
};

//====================================================================

class C : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( fTag_ );
    exprDeps.requires_expression( gTag_ );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    const Expr::FieldMgrSelector<SpatialOps::SingleValueField>::type& fmgr = fml.field_manager<SpatialOps::SingleValueField>();
    f_ = &fmgr.field_ref( fTag_ );
    g_ = &fmgr.field_ref( gTag_ );
  }

  void evaluate() {
    using namespace SpatialOps;
    this->value() <<= 3.3 + *f_ + *g_;
  }

  ~C(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const
    { return new C(fTag_,gTag_); }
    Builder( const Expr::Tag& cTag,
             const Expr::Tag& fTag,
             const Expr::Tag& gTag )
    : ExpressionBuilder(cTag),
      fTag_(fTag), gTag_(gTag)
    {}
    ~Builder(){}
  private:
    const Expr::Tag fTag_, gTag_;
  };

protected:
  C( const Expr::Tag& fTag,
     const Expr::Tag& gTag )
  : Expr::Expression<SpatialOps::SingleValueField>(),
      fTag_(fTag), gTag_(gTag)
  {
    this->set_gpu_runnable(true);
  }
  const Expr::Tag fTag_, gTag_;
  const SpatialOps::SingleValueField *f_, *g_;
};

//====================================================================

class D : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( fTag_ );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    f_ = &fml.field_ref<SpatialOps::SingleValueField>( fTag_ );
  }

  void evaluate(){
    using namespace SpatialOps;
    this->value() <<= 4.4 + *f_;
  }

  ~D(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const
    { return new D(ft_); }
    Builder( const Expr::Tag& dTag,
             const Expr::Tag& ftag )
    : ExpressionBuilder(dTag),
      ft_(ftag) {}
    ~Builder(){}
  private:
    const Expr::Tag ft_;
  };

protected:
  D( const Expr::Tag& fTag )
    : Expr::Expression<SpatialOps::SingleValueField>(),
      fTag_(fTag)
  {
    this->set_gpu_runnable(true);
  }
  const Expr::Tag fTag_;
  const SpatialOps::SingleValueField *f_;
};

//====================================================================

class E : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps ){}
  void bind_fields( const Expr::FieldManagerList& fml ){}
  void evaluate(){
    using namespace SpatialOps;
    this->value() <<= 5.5;
  }

  ~E(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const{ return new E(); }
    Builder( const Expr::Tag& eTag ) : ExpressionBuilder(eTag) {}
    ~Builder(){}
  private:
  };

protected:
  E() : Expr::Expression<SpatialOps::SingleValueField>() 
  {
    this->set_gpu_runnable(true);
  }
};

//====================================================================

class F : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( gTag_ );
    exprDeps.requires_expression( hTag_ );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    const Expr::FieldMgrSelector<SpatialOps::SingleValueField>::type& fmgr = fml.field_manager<SpatialOps::SingleValueField>();
    g_ = &fmgr.field_ref( gTag_ );
    h_ = &fmgr.field_ref( hTag_ );
  }

  void evaluate() {
    using namespace SpatialOps;
    this->value() <<= 6.6 + *g_ + *h_;
  }

  ~F(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const
    { return new F(gTag_,hTag_); }
    Builder( const Expr::Tag& fTag,
             const Expr::Tag& gTag,
             const Expr::Tag& hTag)
    : ExpressionBuilder(fTag),
      gTag_(gTag), hTag_(hTag) {}
    ~Builder(){}
  private:
    const Expr::Tag gTag_, hTag_;
  };

protected:
  F( const Expr::Tag& gTag,
     const Expr::Tag& hTag )
    : Expr::Expression<SpatialOps::SingleValueField>(),
      gTag_(gTag), hTag_(hTag)
  {
    this->set_gpu_runnable(true);
  }
  const Expr::Tag gTag_, hTag_;
  const SpatialOps::SingleValueField *g_, *h_;
};

//====================================================================

class G : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps ){}
  void bind_fields( const Expr::FieldManagerList& fml ){}
  void evaluate() {
    using namespace SpatialOps;
    this->value() <<= 7.7;
  }

  ~G(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const{ return new G(); }
    Builder( const Expr::Tag& gTag ) : ExpressionBuilder(gTag) {}
    ~Builder(){}
  private:
  };

protected:
  G() : Expr::Expression<SpatialOps::SingleValueField>()
  {
    this->set_gpu_runnable(true);
  }
};

//====================================================================

class H : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps ){}
  void bind_fields( const Expr::FieldManagerList& fml ){}
  void evaluate() {
    using namespace SpatialOps;
    this->value() <<= 8.8;
  }

  ~H(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const{ return new H(); }
    Builder( const Expr::Tag& hTag ) : ExpressionBuilder(hTag) {}
    ~Builder(){}
  private:
  };

protected:
  H() : Expr::Expression<SpatialOps::SingleValueField>()
  {
    this->set_gpu_runnable(true);
  }
};

//====================================================================

class I : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps ){
    exprDeps.requires_expression( dTag_ );
  }

  void bind_fields( const Expr::FieldManagerList& fml ){
    d_ = &fml.field_ref<SpatialOps::SingleValueField>( dTag_ );
  }

  void evaluate() {
    using namespace SpatialOps;
    this->value() <<= 9.9 * *d_;
  }

  ~I(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const{ return new I(dT_); }
    Builder( const Expr::Tag& iTag, const Expr::Tag& dTag )
    : ExpressionBuilder(iTag),
      dT_(dTag)
    {}
    ~Builder(){}
  private:
    const Expr::Tag dT_;
  };

protected:
  I( const Expr::Tag dTag )
    : Expr::Expression<SpatialOps::SingleValueField>(),
      dTag_( dTag )
  {
    this->set_gpu_runnable(true);
  }
  const Expr::Tag dTag_;
  const SpatialOps::SingleValueField *d_;
};

//====================================================================
