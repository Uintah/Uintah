#ifndef CPD_Data_h
#define CPD_Data_h

#include <vector>
#include <map>
#include <string>

#include <CCA/Components/Wasatch/Coal/CoalData.h>
#include <CCA/Components/Wasatch/Coal/SpeciesData.h>

namespace CPD{

  /**
   *  \ingroup CPD
   *  \enum CPDSpecies
   *  \brief Enumerates the gas phase species produced by the CPD model.
   */

enum CPDSpecies {
   CO2  = 0,
   H2O  = 1,
   CO   = 2,
   HCN  = 3,
   NH3  = 4,
   CH4  = 5, 
   H    = 6,
   INVALID_SPECIES = 99
   
 };

  /**
   *  \ingroup CPD
   *  \class CPDInformation
   *
   *  \brief This class, provide the functional group data, molecular
   *         weight , amount of laible bridge, for each type of coal.
   *
   */
  class CPDInformation{

  public:	

    /**
     *  \brief Construct a CPDInformation class
     *  \param coalType the CoalType
     */
    CPDInformation( const Coal::CoalType coalType );

    int get_nspec() const{ return nspec_; } ///< Return number of gas-phase species for the CPD model

    const Coal::CoalComposition& get_coal_composition() const{ return coalComp_; } ///< return the CoalComposition

    const std::vector<double>& get_fgi() const{ return fg_; } ///< Functional Group vector (mole basis)

    const std::vector<double>& get_mwVec() const{ return mwVec_;};

    double get_sumfg() const{ return sumfg_; };  ///< Sum of Functional Groups (mole basis)

    double get_l0_mass() const { return l0_; };  ///< Initial mass fraction of labile bridge

    double get_l0_mole() const; ///< initial mole fraction of labile bridge in coal
    
    double get_tarMonoMW() const { return coalComp_.get_tarMonoMW(); }; ///< tar monomer molecular weight
    
    const double get_tarMassFrac() const{ return tarMassFrac0_; } ///< Functional Group vector (mole basis)

    double get_coordNo() const { return coordNo_; }; ///< coordination number of coal lattice
   
    double get_lbPop0() const { return lbPop0_; }; ///< initial normalized bridge population

    const std::vector<double>& get_A0() const { return A0_; }; ///< Pre exponential factor of devolatilization reaction

    const std::vector<double>& get_E0() const { return E0_; }; ///< Activation energy of reaction of devolatilization 

    const std::vector<double>& get_sigma() const {return sigma_;} ///< deviation of activation energy 

    double get_hypothetical_volatile_mw() const{ return Mw_; }

    double l0_molecular_weight() const{ return Ml0_; }
    

  protected:
    const int nspec_;                      ///< the number of gas phase species
    const Coal::CoalType coalType_;        ///< the CoalType
    const Coal::CoalComposition coalComp_;
    const double l0_;
    const double Ml0_;                   ///<  Mole fraction of labile bridge
    double Mw_;                          ///<  Molecular weight of hypothetical carbon
    double coordNo_;                     ///<  coordination number for coal lattice
    double lbPop0_;                      ///<  initial normalized labile bridge population in coal
    double tarMassFrac0_;                ///<  initial mass fraction of tar in coal.

    std::vector<double> fg_;          ///< mass fractions of fgs in coal
    double sumfg_;           ///< Sum of fgs (mass basis): this includes tarFrac0_
    std::vector<double> mwVec_;
    std::vector<double> A0_;
    std::vector<double> E0_;
    std::vector<double> sigma_;

    GasSpec::SpeciesData speciesData_;
  };


  typedef std::vector<int> VecI;
  typedef std::pair< double, VecI > MWCompPair; 
  typedef std::vector<MWCompPair> SpecContributeVec;

  /**
   *  \ingroup CPD
   *  \class SpeciesSum
   *  \brief Holds information about the contribution of each gi to
   *         the species compositions
   */
  class SpeciesSum
  {
    SpeciesSum();
    SpeciesSum& operator=( const SpeciesSum& );  // no assignment
    SpeciesSum( const SpeciesSum& );  // no copying
  public:

    static const SpeciesSum& self();

    ~SpeciesSum();
    // obtain a vector of vector which shows the contribution of each species to a component.
    const SpecContributeVec& get_vec_comp() const{ return sContVec_; } 

    // obtain number of components
    int get_ncomp() const{ return sContVec_.size(); }
    
  private:
    SpecContributeVec sContVec_;
    MWCompPair species_connect( const CPDSpecies spec );
  };

  std::vector<double> deltai_0( const CPDInformation& info, const double c0 );
  double tar_0(const CPDInformation& info, const double c0);


} // namespace CPD

#endif // CPD_Data_h
