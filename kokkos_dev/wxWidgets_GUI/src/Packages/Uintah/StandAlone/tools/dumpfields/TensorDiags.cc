#include "TensorDiags.h"

#include <string>
#include <map>
#include <list>

using namespace std;

namespace Uintah {
  using namespace SCIRun;
  
  TensorDiag::~TensorDiag() {}
  
  // --------------------------------------------------------------------------
  
  class TensorValueDiag : public TensorDiag {
  public:
    TensorValueDiag() {}
    string name() const { return "value"; }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    NCVariable<Matrix3> & res) const {
      da->query(res, fieldname, imat, patch, time);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    CCVariable<Matrix3> & res) const {
      da->query(res, fieldname, imat, patch, time);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    ParticleSubset * parts,
                    ParticleVariable<Matrix3> & res) const {
      da->query(res, fieldname, imat, patch, time);
    }
  };
  
  // --------------------------------------------------------------------------
  
  class TensorToTensorDiag : public TensorDiag {
  public:
    TensorToTensorDiag() {}
    virtual ~TensorToTensorDiag() {}
        
    virtual Matrix3 convert(const Matrix3 & val) const = 0;
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    NCVariable<Matrix3> & res) const {
      NCVariable<Matrix3> value;
      da->query(value, fieldname, imat, patch, time);
      res.allocate(value.getLowIndex(), value.getHighIndex());
      for(NCVariable<Matrix3>::iterator it=value.begin();it!=value.end();it++)
        res[it.getIndex()] = convert(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    CCVariable<Matrix3> & res) const {
      CCVariable<Matrix3> value;
      da->query(value, fieldname, imat, patch, time);
      res.allocate(value.getLowIndex(), value.getHighIndex());
      for(CCVariable<Matrix3>::iterator it=value.begin();it!=value.end();it++)
        res[it.getIndex()] = convert(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    ParticleSubset * parts,
                    ParticleVariable<Matrix3> & res) const {
      ParticleVariable<Matrix3> value;
      da->query(value, fieldname, imat, patch, time);
      res.allocate(parts);
      for(ParticleSubset::iterator pit(parts->begin());pit!=parts->end();pit++)
        res[*pit] = convert(value[*pit]);
    }
  };
  
  // --------------------------------------------------------------------------
  
  class PreTensorToTensorDiag : public TensorDiag {
  public:
    PreTensorToTensorDiag(const TensorDiag * preop_, const TensorToTensorDiag * realdiag_)
      : preop(preop_), realdiag(realdiag_) {}
    
    ~PreTensorToTensorDiag() {}
    
    std::string name() const { return preop->name()+"_"+realdiag->name(); }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    NCVariable<Matrix3>  & res) const {
      NCVariable<Matrix3> fullvals;
      (*preop)(da, patch, fieldname, imat, time, fullvals);
      res.allocate(fullvals.getLowIndex(), fullvals.getHighIndex());
      for(NCVariable<Matrix3>::iterator it=fullvals.begin();it!=fullvals.end();it++)
        res[it.getIndex()] = realdiag->convert(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    CCVariable<Matrix3>  & res) const {
      NCVariable<Matrix3> fullvalues;
      (*preop)(da, patch, fieldname, imat, time, fullvalues);
      res.allocate(fullvalues.getLowIndex(), fullvalues.getHighIndex());
      for(CCVariable<Matrix3>::iterator it=fullvalues.begin();it!=fullvalues.end();it++)
        res[it.getIndex()] = realdiag->convert(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time,
                    ParticleSubset * pset,
                    ParticleVariable<Matrix3> & res) const {
      ParticleVariable<Matrix3> fullvalues;
      fullvalues.allocate(pset);
      (*preop)(da, patch, fieldname, imat, time, pset, fullvalues);
      res.allocate(pset);
      for(ParticleSubset::iterator pit(pset->begin());pit!=pset->end();pit++)
        res[*pit] = realdiag->convert( fullvalues[*pit] );
    }
    
  private:
    const TensorDiag         * preop;
    const TensorToTensorDiag * realdiag;
  };
  
  // --------------------------------------------------------------------------
  
  class Matrix3ValueDiag : public TensorToTensorDiag {
  public:
    std::string name() const { return "value"; }
    Matrix3 convert(const Matrix3 & a) const {
      return a;
    }
  };
  
  class Matrix3GreenStrainDiag : public TensorToTensorDiag {
  public:
    std::string name() const { return "green_strain"; }
    Matrix3 convert(const Matrix3 & a) const {
      Matrix3 b(-0.5,0,0, 0,-0.5,0, 0,0,-0.5);
      for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)
          b(i,j) += 0.5*a(k,i)*a(k,j);
      return b;
    }
  };
  
  class Matrix3RightCGDiag : public TensorToTensorDiag {
  public:
    std::string name() const { return "right_cg"; }
    Matrix3 convert(const Matrix3 & a) const {
      Matrix3 b(0,0,0, 0,0,0, 0,0,0);
      for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)
          b(i,j) += a(k,i)*a(k,j);
      return b;
    }
  };
  
  class Matrix3LogarithmicDiag : public TensorToTensorDiag {
  public:
    std::string name() const { return "logarithmic"; }
    Matrix3 convert(const Matrix3 & a) const {
      Matrix3 b(0,0,0, 0,0,0, 0,0,0);
      for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
        for(int k=0;k<3;k++)
          b(i,j) += a(k,i)*a(k,j);
        b(i,j) = 0.5*log(b(i,j));
      }
      return b;
    }
  };
  
  class Matrix3DeviatorDiag : public TensorToTensorDiag {
  public:
    std::string name() const { return "deviator"; }
    Matrix3 convert(const Matrix3 & a) const {
      Matrix3 b(0,0,0, 0,0,0, 0,0,0);
      
      double P = 0.;
      for(int i=0;i<3;i++) P += -a(i,i)/3.0;
      
      for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
        b(i,j) = a(i,j) + P;
      }
      return b;
    }
  };
  
  // --------------------------------------------------------------------------
  
  
  typedef vector<TensorDiag *>     TDiagTable;
  typedef map<string, TDiagTable > PreTDiagTable;
  
  TDiagTable    _ttdiagtable;
  PreTDiagTable _ttprediagtable;
  
  void createTensorDiags() {
    if(_ttdiagtable.size()) return;
    
    _ttdiagtable.push_back(new Matrix3ValueDiag());
    _ttdiagtable.push_back(new Matrix3GreenStrainDiag());
    _ttdiagtable.push_back(new Matrix3RightCGDiag());
    _ttdiagtable.push_back(new Matrix3LogarithmicDiag());
    _ttdiagtable.push_back(new Matrix3DeviatorDiag());
  }
  
  void destroyTensorDiags() {
    for(vector<TensorDiag *>::iterator dit(_ttdiagtable.begin());dit!=_ttdiagtable.end();dit++) {
        delete *dit;
      }
    _ttdiagtable.clear();
  }
  
  void describeTensorDiags(ostream & os)
  {
    createTensorDiags();
    os << "  Tensor -> Tensor" << endl;
    for(vector<TensorDiag *>::iterator dit(_ttdiagtable.begin());
        dit!=_ttdiagtable.end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }
  }
  
  // --------------------------------------------------------------------------
  
  int numberOfTensorDiags(const TypeDescription * fldtype) {
    createTensorDiags();
    if(fldtype->getSubType()->getType()!=TypeDescription::Matrix3) return 0;
    return _ttdiagtable.size();
  }
  
  std::string tensorDiagName(const TypeDescription * fldtype, int idiag) {
    createTensorDiags();
    return _ttdiagtable[idiag]->name();
  }
  
  TensorDiag const * createTensorDiag(const TypeDescription * fldtype, int idiag,
                                      const TensorDiag * tensorpreop)
  {
    createTensorDiags();

    TensorDiag const * res = _ttdiagtable[idiag];
    TensorToTensorDiag const * ttres = dynamic_cast<TensorToTensorDiag const *>(res);
    if(tensorpreop &&  ttres) {
      if(!_ttprediagtable.count(tensorpreop->name()))
        _ttprediagtable[ tensorpreop->name() ].resize( _ttprediagtable.size() );
      
      if(!_ttprediagtable[ tensorpreop->name() ][idiag] )
        _ttprediagtable[ tensorpreop->name() ][idiag] = new PreTensorToTensorDiag( tensorpreop, ttres );
      
      return _ttprediagtable[ tensorpreop->name() ][idiag];
    }
    return res;
  }
  
  TensorDiag const *
  createTensorOp(const FieldSelection & fldselection)
  {
    createTensorDiags();
    
    TensorDiag const *  res = 0;
    size_t ndiags = _ttdiagtable.size();
    for(size_t idiag=0;idiag<ndiags;idiag++)
      {
        if (fldselection.wantTensorOp(_ttdiagtable[idiag]->name()))
          {
            res = _ttdiagtable[idiag];
          }
      }
    return res;
  }
  
  list<TensorDiag const *> 
  createTensorDiags(const TypeDescription * fldtype, 
                    const FieldSelection & fldselection,
                    const TensorDiag * preop)
  {
    list<TensorDiag const *>  res;
    int ndiags = numberOfTensorDiags(fldtype);
    for(int idiag=0;idiag<ndiags;idiag++)
      {
        if(fldselection.wantDiagnostic(tensorDiagName(fldtype, idiag))) {
          res.push_back( createTensorDiag(fldtype, idiag, preop) );
        }
      }
    
    return res;
  }
  
}
