#ifndef POUnet_h
#define POUnet_h


#include <spatialops/Nebo.h>
#include <expression/Expression.h>


// ###################################################################
//
//                    Functions for reading files
//
// ###################################################################
namespace POU{

void assert_line_name(std::string line_name, std::ifstream& file, std::string line){
  std::getline(file, line);
  if (line!=line_name){throw std::runtime_error("Did not find "+line_name);}
}

template<typename T>
T read_line_into_value(std::ifstream& file, std::string line)
{
  T result;
  std::getline(file, line);
  std::istringstream iss(line);
  iss >> result;
  return result;
}

template<typename T>
std::vector<T> read_lines_into_vector(std::ifstream& file, std::string line, int nlines)
{
  std::vector<T> result(nlines);
  for (int i=0; i<nlines; i++){
    result[i] = POU::read_line_into_value<T>(file, line);
  }
  return result;
}

}

// ###################################################################
//
//            POUnetData - for reading single property
//
// ###################################################################

class POUnetData{
public:
  void read_from_file(std::ifstream & file, std::string line){
    POU::assert_line_name("ndim", file, line);
    ndim_ = POU::read_line_into_value<unsigned int>(file, line);

    POU::assert_line_name("npartition", file, line);
    npartition_ = POU::read_line_into_value<unsigned int>(file, line);

    POU::assert_line_name("nbasis", file, line);
    nbasis_ = POU::read_line_into_value<unsigned int>(file, line);

    POU::assert_line_name("float_type", file, line);
    std::string dtype = POU::read_line_into_value<std::string>(file, line);  // read but don't need in C++

    POU::assert_line_name("transform_power", file, line);
    transform_power_ = POU::read_line_into_value<double>(file, line);

    POU::assert_line_name("transform_shift", file, line);
    transform_shift_ = POU::read_line_into_value<double>(file, line);

    POU::assert_line_name("transform_sign_shift", file, line);
    transform_sign_shift_ = POU::read_line_into_value<double>(file, line);

    POU::assert_line_name("ivar_center", file, line);
    ivar_center_ = POU::read_lines_into_vector<double>(file, line, ndim_);

    POU::assert_line_name("ivar_scale", file, line);
    ivar_scale_ = POU::read_lines_into_vector<double>(file, line, ndim_);

    for(unsigned int i=0; i<ndim_; i++){
      ivar_invscale_.push_back(1./ivar_scale_[i]);
    }

    unsigned int npartition_params = ndim_*npartition_;
    unsigned int ncoeffs = nbasis_*ndim_+1;
    ncoeffs += (ndim_*(ndim_+1)*0.5-ndim_);
    ncoeffs *= npartition_;

    POU::assert_line_name("centers", file, line);
    centers_ = POU::read_lines_into_vector<double>(file, line, npartition_params);

    POU::assert_line_name("shapes", file, line);
    shapes_ = POU::read_lines_into_vector<double>(file, line, npartition_params);

    POU::assert_line_name("coeffs", file, line);
    coeffs_ = POU::read_lines_into_vector<double>(file, line, ncoeffs);

    totalbasis_ = 1 + ndim_*nbasis_;
    if (nbasis_>1){
      totalbasis_ += (ndim_*(ndim_+1)*0.5-ndim_);
    }
  };

  ~POUnetData(){};

  unsigned int ndim_;
  unsigned int npartition_;
  unsigned int nbasis_;
  unsigned int totalbasis_;

  double transform_power_;
  double transform_shift_;
  double transform_sign_shift_;
  std::vector<double> ivar_center_;
  std::vector<double> ivar_scale_;
  std::vector<double> ivar_invscale_;

  std::vector<double> centers_;
  std::vector<double> shapes_;
  std::vector<double> coeffs_;
};

// ###################################################################
//
//                    POUnet Model - single property
//
// ###################################################################

class POUnet{
public:
  POUnet(const POUnetData & pd):
    ndim_(pd.ndim_),
    npartition_(pd.npartition_),
    nbasis_(pd.nbasis_),
    totalbasis_(pd.totalbasis_),
    transform_inv_power_(1./pd.transform_power_),
    transform_shift_(pd.transform_shift_),
    transform_sign_shift_(pd.transform_sign_shift_),
    ivar_center_(pd.ivar_center_),
    ivar_invscale_(pd.ivar_invscale_),
    centers_(pd.centers_),
    invshapes_(pd.shapes_),
    coeffs_(pd.coeffs_),
    query_buffer_(ndim_)
  { }

  ~POUnet(){};

  double query( const double* const indep ) const{
    // center/scale inputs
    for (int i=0; i<ndim_; i++){
      query_buffer_[i] = (indep[i] - ivar_center_[i])*ivar_invscale_[i];
    }

    // evaluate basis functions
    std::vector<double> basis_out(totalbasis_, 1.); // constant
    if (nbasis_>0){ // linear
      for (int i=0; i<ndim_; i++){
        basis_out[i+1] = query_buffer_[i];
      }
    }
    if (nbasis_>1){ // quadratic
      for (int i=0; i<ndim_; i++){
        basis_out[i+1+ndim_] = query_buffer_[i]*query_buffer_[i];
      }
      if (ndim_>1){
        basis_out[1+ndim_*2] = query_buffer_[0]*query_buffer_[1];
      }
      if (ndim_>2){
        basis_out[1+ndim_*2+1] = query_buffer_[0]*query_buffer_[2];
        basis_out[1+ndim_*2+2] = query_buffer_[1]*query_buffer_[2];
      }
      if (ndim_>3){throw std::runtime_error("Unsupported cross terms.");}
    }
    if (nbasis_>2){throw std::runtime_error("Unsupported degree.");}

    double partition_sum = 0.;
    double output = 0.;

    for (int i=0; i<npartition_; i++){
      double partition_out = 0.;

      // evaluate partitions
      for(int j=0; j<ndim_; j++){
        const double preexp = (query_buffer_[j] - centers_[i*ndim_+j])*invshapes_[i*ndim_+j];
        partition_out += preexp * preexp;
      }
      partition_out = std::exp(-partition_out);
      partition_sum += partition_out; // keep track of sum for normalizing

      for(int k=0; k<totalbasis_; k++){
        output += partition_out*basis_out[k]*coeffs_[i*totalbasis_+k]; // multiply partition by basis
      }
    }
    output /= partition_sum; // normalize partitions by sum
    double t_output = transform(output);
    return t_output;
  };

private:
    const unsigned int ndim_;
    const unsigned int npartition_;
    const unsigned int nbasis_;
    const unsigned int totalbasis_;

    const double transform_inv_power_;
    const double transform_shift_;
    const double transform_sign_shift_;
    const std::vector<double> ivar_center_;
    const std::vector<double> ivar_invscale_;

    const std::vector<double> centers_;
    const std::vector<double> invshapes_;
    const std::vector<double> coeffs_;

    mutable std::vector<double> query_buffer_;

    double transform(const double input)const{
      if (transform_inv_power_==1. && transform_shift_==0. && transform_sign_shift_==0.){
        return input;
      }
      else{
        double offset = (input<0.) ? -transform_sign_shift_ : transform_sign_shift_;
        double shift = input - offset;
        double pow_shift = std::pow(std::abs(shift), transform_inv_power_);
        double sign_pow_shift = (shift<0.) ? -pow_shift : pow_shift;
        return sign_pow_shift - transform_shift_;
      }
    }
};

// ###################################################################
//
//                 POUnet Manager - all Properties
//
// ###################################################################

class POUnetManager{
public:
  POUnetManager(const std::string folder)
  {
    std::ifstream ifile(folder+"ivar_names.txt");
    std::string iline;
    if (ifile.is_open()){
      while(std::getline(ifile, iline)){
        ivar_names_.push_back(iline);
      }
    }
    else{
      throw std::runtime_error("could not find ivar_names.txt in "+folder);
    }

    std::ifstream dfile(folder+"dvar_names.txt");
    std::string dline;
    if (dfile.is_open()){
      while(std::getline(dfile, dline)){
        dvar_names_.push_back(dline);
      }
    }
    else{
      throw std::runtime_error("could not find dvar_names.txt in "+folder);
    }

    // loop over dvars and create/store models
    for (int i=0; i<dvar_names_.size(); i++){
      std::string txtfile = folder + dvar_names_[i] + ".txt";
      POUnetData pd;
      std::string line;
      std::ifstream file(txtfile);
      if(file.is_open()){
        std::getline(file, line);
        while (line!="POUNET"){std::getline(file, line);}
      }
      pd.read_from_file(file, line);

      if (pd.ndim_!=ivar_names_.size()){throw std::runtime_error("Inconsistent input dimensionality for "+dvar_names_[i]);}

      auto temp_model_pointer = std::make_shared<POUnet>(pd);
      models_.insert(std::make_pair(dvar_names_[i], temp_model_pointer));
    }
  }

  ~POUnetManager(){};

  const std::vector<std::string> get_indepvar_names() const{
    return ivar_names_;
  }
  
  const std::vector<std::string> get_depvar_names() const{
    return dvar_names_;
  }

  bool has_depvar( const std::string name ) const{
    if (std::find(dvar_names_.begin(), dvar_names_.end(), name) != dvar_names_.end()){ return true; }
    return false;
  }

  std::vector<std::string> ivar_names_;
  std::vector<std::string> dvar_names_;
  std::map<std::string, std::shared_ptr<POUnet>> models_;
};

// ###################################################################
//
//                             Evaluator
//
// ###################################################################


template< typename FieldT >
class POUnetEvaluator
 : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS( FieldT, indepVars_ )
  std::shared_ptr<const POUnet> evaluator_;

  POUnetEvaluator( std::shared_ptr<const POUnet> model,
                   const Expr::TagList& ivarNames );

  class Functor{
    const POUnet& eval_;
  public:
    Functor( const POUnet& model ) : eval_(model){}
    double operator()( const double x ) const{
      return eval_.query(&x);
    }
    double operator()( const double x1, const double x2 ) const{
      double vals[2] = {x1,x2};
      return eval_.query(vals);
    }
    double operator()( const double x1, const double x2, const double x3 ) const{
      double vals[3] = {x1,x2,x3};
      return eval_.query(vals);
    }
  };

public:
  class Builder : public Expr::ExpressionBuilder
  {
    std::shared_ptr<const POUnet> model_;
    const Expr::TagList ivarNames_;
  public:
    Builder( const Expr::Tag& result,
             std::shared_ptr<const POUnet> model,
             const Expr::TagList& ivarNames )
      : ExpressionBuilder(result),
        model_( model ),
        ivarNames_( ivarNames )
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new POUnetEvaluator<FieldT>( model_, ivarNames_ );
    }
  };

  ~POUnetEvaluator();

  void evaluate();
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################


template< typename FieldT >
POUnetEvaluator<FieldT>::
POUnetEvaluator( std::shared_ptr<const POUnet> model,
                 const Expr::TagList& ivarNames )
  : Expr::Expression<FieldT>(),
    evaluator_( model )
{
//   this->set_gpu_runnable( true ); // apply_pointwise is not GPU ready.
  this->template create_field_vector_request<FieldT>(ivarNames, indepVars_);
}

//--------------------------------------------------------------------

template< typename FieldT >
POUnetEvaluator<FieldT>::
~POUnetEvaluator()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
POUnetEvaluator<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  switch( indepVars_.size() ){
    case 1:{
      result <<= apply_pointwise( factory<Functor>(*evaluator_),
                                  indepVars_[0]->field_ref() );
      break;
    }
    case 2:{
      result <<= apply_pointwise( factory<Functor>(*evaluator_),
                                  indepVars_[0]->field_ref(),
                                  indepVars_[1]->field_ref() );
      break;
    }
    case 3:{
      result <<= apply_pointwise( factory<Functor>(*evaluator_),
                                  indepVars_[0]->field_ref(),
                                  indepVars_[1]->field_ref(),
                                  indepVars_[2]->field_ref() );
      break;
    }
    default:
      throw std::invalid_argument( "Unsupported dimensionality for POUnet" );
  }
}

//--------------------------------------------------------------------

#endif // POUnet_h
