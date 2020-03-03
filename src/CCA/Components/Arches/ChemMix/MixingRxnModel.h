/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- MixingRxnModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MixingRxnModel_h
#define Uintah_Component_Arches_MixingRxnModel_h

// Uintah includes
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DebugStream.h>

// C++ includes
#include     <vector>
#include     <map>
#include     <string>
#include     <stdexcept>
#include <sci_defs/kokkos_defs.h>

#define ALMOST_A_MAGIC_NUMBER 3


/**
 * @class  MixingRxnModel
 * @author Charles Reid
 * @date   Nov, 22 2008
 *
 * @brief Base class for mixing/reaction tables interfaces
 *
 *
 This MixingRxnModel class provides a representation of the mixing
 and reaction model for the Arches code (interfaced through Properties.cc).
 The MixingRxnModel class is a base class that allows for child classes
 that each provide a specific representation of specific mixing and
 reaction table formats.

 Tables are pre-processed by using any number of programs (DARS, Cantera, TabProps,
 etc.).
 *
 */


namespace Uintah {

  // setenv SCI_DEBUG TABLE_DEBUG:+
  static DebugStream cout_tabledbg("TABLE_DEBUG",false);

  class MixingRxnModel{

  public:

    // Useful typedefs
    typedef std::map<std::string, const VarLabel* >           VarMap;
    typedef std::map<std::string, CCVariable<double>* >       CCMap;
    typedef std::map<std::string, double >                    doubleMap;
    typedef std::map<std::string, doubleMap>                  InertMasterMap;
    struct ConstVarContainer {

      constCCVariable<double> var;

    };
    typedef std::map<std::string, ConstVarContainer> StringToCCVar;

    MixingRxnModel( MaterialManagerP& materialManager );

    virtual ~MixingRxnModel();

    /** @brief Interface the the input file.  Get table name, then read table data into object */
    virtual void problemSetup( const ProblemSpecP& params ) = 0;

    /** @brief Returns a vector of the state space for a given set of independent parameters */
    virtual void sched_getState( const LevelP& level,
                                 SchedulerP& sched,
                                 const int time_substep,
                                 const bool initialize,
                                 const bool modify_ref_den ) = 0;

    /** @brief Provides access for models, algorithms, etc. to add additional table lookup variables. */
    void addAdditionalDV( std::vector<std::string>& vars );

    /** @brief Returns the value of a single variable given the iv vector
     * This will be useful for other classes to have access to */
    virtual double getTableValue( std::vector<double>, std::string ) = 0;

    /** @brief Get a table value **/
    virtual double getTableValue( std::vector<double> iv, std::string depend_varname,
        StringToCCVar inert_mixture_fractions, IntVector c) = 0;

    /** @brief Get a table value **/
    virtual double getTableValue( std::vector<double> iv, std::string depend_varname,
                   doubleMap inert_mixture_fractions, bool do_inverse = false ) = 0;

    /** @brief For efficiency: Matches tables lookup species with pointers/index/etc */
    virtual void tableMatching() = 0;

    /** @brief Return a reference to the independent variables */
    inline const VarMap getIVVars(){ return d_ivVarMap; };

    /** @brief Return a reference to the dependent variables */
    inline const VarMap getDVVars(){ return d_dvVarMap; };

    /** @brief Return a string list of all independent variable names in order */
    inline std::vector<std::string>& getAllIndepVars(){ return d_allIndepVarNames; };

    /** @brief Return a string list of dependent variables names in the order they were read */
    inline std::vector<std::string>& getAllDepVars(){ return d_allDepVarNames; };

    /** @brief Returns a <string, double> map of KEY constants found in the table */
    inline doubleMap& getAllConstants(){ return d_constants; };

    inline double getDoubleTableConstant(const std::string key ){

      doubleMap::iterator iter = d_constants.find(key);

      if ( iter != d_constants.end() ){
        return iter->second;
      } else {
        throw InvalidValue("Error: Table constant not found: "+key,__FILE__,__LINE__);
      }
    }

    /** @brief Returns the map of participating inerts **/
    inline InertMasterMap& getInertMap(){ return d_inertMap; };

    /** @brief Returns a boolean regarding if post mixing is used or not **/
    inline bool doesPostMix(){ return d_does_post_mixing; };

    /** @brief  Insert the name of a dependent variable into the dependent variable map (dvVarMap),
                which maps strings to VarLabels */
    inline bool insertIntoMap( const std::string var_name ){

      // Check to ensure this variable is in this table:
      auto i_var = std::find( d_allDepVarNames.begin(), d_allDepVarNames.end(), var_name );

      if ( i_var != d_allDepVarNames.end() ){

        VarMap::iterator i = d_dvVarMap.find( var_name );

        if ( i == d_dvVarMap.end() ) {

          const VarLabel* the_label = VarLabel::create( var_name, CCVariable<double>::getTypeDescription() );
          d_dvVarMap.insert( std::make_pair( var_name, the_label ) );

        }

        return true;

      } else {

        return false;

      }

    };

    /** @brief  Insert the name of a dependent variable into the dependent variable map (dvVarMap),
                which maps strings to VarLabels */
    inline bool insertOldIntoMap( const std::string var_name ){

      // Check to ensure this variable is in this table:
      auto i_var = std::find( d_allDepVarNames.begin(), d_allDepVarNames.end(), var_name );

      if ( i_var != d_allDepVarNames.end() ){

        VarMap::iterator i = d_oldDvVarMap.find( var_name );

        if ( i == d_oldDvVarMap.end() ) {

          std::string name = var_name+"_old";

          const VarLabel* the_old_label = VarLabel::create( name, CCVariable<double>::getTypeDescription() );
          d_oldDvVarMap.insert( std::make_pair( name, the_old_label ) );

        }

        return true;

      } else {

        return false;

      }
    };

    inline double get_Ha( std::vector<double>& iv, double inerts ){

      double ha = _iv_transform->get_adiabatic_enthalpy( iv, inerts );
      return ha;

    };

    inline double get_reference_density(CCVariable<double>& density, constCCVariable<double> volFraction ){

      if ( d_user_ref_density ){
        return d_reference_density;
      } else {
        if ( volFraction[d_ijk_den_ref] > .5 ){
          return density[d_ijk_den_ref]; }
        else
          throw InvalidValue("Error: Your reference density is in a wall. Choose another reference location.",
                             __FILE__,__LINE__);
      }

    };

    /** @brief Get a dependant variable's index **/
    virtual int findIndex(std::string) = 0;

    /** @brief Returns false if flow is changing temperature */
    // In other words, is temperature changing?
    // The strange logic is to make the logic less confusing in Properties.cc
    bool is_not_cold(){ if (d_coldflow){ return false; }else{ return true; }};

  protected :

    struct DepVarCont {

      CCVariable<double>* var;
      int index;

    };

    std::string _temperature_label_name;

    IntVector d_ijk_den_ref;                      ///< Reference density location

    class TransformBase {

      public:
        TransformBase();
        virtual ~TransformBase();

        /** @brief Interface to the input file **/
        virtual bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ) = 0;
        /** @brief Transforms the mixture fractions that are transported to the lookup IVs **/
        virtual void transform( std::vector<double>& iv, double inert ) = 0;
        /** @brief AFTER the transform has occured, returns the two stream mixture fraction f = primary / ( primary + secondary ).
                   Warning: This might not be general for case which have more than two streams. **/
        virtual double get_adiabatic_enthalpy( std::vector<double>& iv, double inert ) = 0;

        /** @brief Get the heat loss upper and lower bounds **/
        virtual const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> size ) = 0;

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        virtual const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<Kokkos::RandomAccess> >  const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> >  const  size )=0;
#endif

        /** @brief Check to see if this table deals with heat loss **/
        virtual bool has_heat_loss() = 0;

        /** @brief Return the independent variable space for the reference point as specified in the input UPS **/
        virtual struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv() = 0;

      protected:
        int _index_1;
        int _index_2;
        int _index_3;
        std::string _index_1_name;
        std::string _index_2_name;
        std::string _index_3_name;

    };

    class NoTransform : public TransformBase {

      public:
        NoTransform();
        ~NoTransform();

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){
          bool no_transform_on = true;
          return no_transform_on;
        };

        void inline transform( std::vector<double>& iv, double inert ){};

        double inline get_adiabatic_enthalpy( std::vector<double>& iv, double inerts ){
          throw InvalidValue("Error: Cannot return an adiabatic enthalpy for this case. Make sure you have identified your table properly (e.g., coal, rcce, standard_flamelet, etc... )",__FILE__,__LINE__);
        };

        bool has_heat_loss(){ return false; };

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        inline const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const  size )
        {
          std::vector<double> hl_bounds;
          hl_bounds.push_back(-1.0);
          hl_bounds.push_back(1.0);
          return hl_bounds;
        };
#endif

        const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> size )
        {
          std::vector<double> hl_bounds;
          hl_bounds.push_back(-1.0);
          hl_bounds.push_back(1.0);
          return hl_bounds;
        };

        struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv(){
          throw InvalidValue("Error: Cannot return the reference values. Make sure you have identified your table properly (e.g., coal, rcce, standard_flamelet, etc... )",__FILE__,__LINE__);
        };
    };

    class SingleIV : public TransformBase {

      //Single IV assumes that the IVs are ordered:
      //(1) mixture fraction (for example)

      public:
        SingleIV( std::map<std::string, double>& keys, MixingRxnModel* const model);
        ~SingleIV();

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool sf_transform = false;
          ProblemSpecP p = ps;

          bool cold_flow;
          p->getWithDefault( "cold_flow",cold_flow,false);

          if ( p->findBlock("single_iv")){

            p->findBlock("single_iv")->getAttribute("iv_label",_f_name);
            sf_transform = true;

          }

          if ( p->findBlock("reference_state") ){
            p->findBlock("reference_state")->getAttribute("iv",_f_ref);
          } else {
            throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
          }

          _f_index = 0;

          if ( !cold_flow ){
            std::vector<double> my_ivs;
            my_ivs.push_back(1);
            double h = _model->getTableValue( my_ivs, "adiabaticenthalpy" );
            _H_fuel = h;

            my_ivs[0] = 0;
            h = _model->getTableValue( my_ivs, "adiabaticenthalpy" );
            _H_ox = h;
          } else {
            _H_ox = 0.0;
            _H_fuel = 0.0;
          }

          return sf_transform;

        };

        bool inline has_heat_loss(){ return false; };

        void inline transform( std::vector<double>& iv, double inert ){
          //do nothing here since there is only one iv.
        };

        double inline get_adiabatic_enthalpy( std::vector<double>& iv, double inerts ){
          return iv[0]*_H_fuel + ( 1.0 - iv[0] )*_H_ox;
        };

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        inline const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const  size )
        {
          std::vector<double> hl_b;
          return hl_b;
        };
#endif

        const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> const size )
        {
          //no heat loss
          std::vector<double> hl_b;
          return hl_b;
        };

        struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv(){
          struct1DArray<double,ALMOST_A_MAGIC_NUMBER> iv(1);
          iv[0]=_f_ref;
          return iv;
        };

      private:

        std::string _f_name;

        double _f_ref;

        std::map<std::string,double> _keys;

        int _f_index;

        double _H_fuel;
        double _H_ox;

        MixingRxnModel* const _model;

    };

    class MFHLTransform : public TransformBase {

      //Double IV assumes that the IVs are ordered:
      //(1) mixture fraction (2) heat loss

      public:

        MFHLTransform( std::map<std::string, double>& keys, MixingRxnModel* const model);
        ~MFHLTransform();

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool sf_transform = false;
          ProblemSpecP p = ps;

          bool cold_flow;
          p->getWithDefault( "cold_flow",cold_flow,false);

          if ( p->findBlock("mixfrac_with_heatloss")){

            p->findBlock("mixfrac_with_heatloss")->getAttribute("f_label", _f_name);
            p->findBlock("mixfrac_with_heatloss")->getAttribute("hl_label",_hl_name);
            sf_transform = true;

          }

          if ( p->findBlock("reference_state") ){
            p->findBlock("reference_state")->getAttribute("f",_f_ref);
            p->findBlock("reference_state")->getAttribute("hl",_hl_ref);
          } else {
            throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
          }

          _f_index = 0;
          _hl_index = 1;

          if ( !cold_flow ){
            std::vector<double> my_ivs;
            my_ivs.push_back(1.0);
            my_ivs.push_back(0.0);
            double h = _model->getTableValue( my_ivs, "adiabaticenthalpy" );
            _H_fuel = h;

            my_ivs[0] = 0;
            my_ivs[1] = 0;
            h = _model->getTableValue( my_ivs, "adiabaticenthalpy" );
            _H_ox = h;
          } else {
            _H_ox = 0.0;
            _H_fuel = 0.0;
          }

          return sf_transform;

        };

        bool inline has_heat_loss(){ return true; };

        void inline transform( std::vector<double>& iv, double inert ){
          //do nothing here since there is only one iv.
        };

        double inline get_adiabatic_enthalpy( std::vector<double>& iv, double inerts ){
          return iv[0]*_H_fuel + ( 1.0 - iv[0] )*_H_ox;
        };

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        inline const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const  size )
        {
          std::vector<double> hl_bounds;

          hl_bounds.push_back(iv_grids(0, 0));
          hl_bounds.push_back(iv_grids(0, size(1)-1));

          return hl_bounds;
        };
#endif

        const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> const size )
        {
          std::vector<double> hl_bounds;

          hl_bounds.push_back(iv_grids[0][0]);
          hl_bounds.push_back(iv_grids[0][size[1]-1]);

          return hl_bounds;
        };

       struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv(){
          struct1DArray<double,ALMOST_A_MAGIC_NUMBER> iv(2);
          iv[0]=_f_ref;
          iv[1]=_hl_ref;
          return iv;
        };

      private:

        std::string _f_name;
        std::string _hl_name;

        double _f_ref;
        double _hl_ref;

        std::map<std::string,double> _keys;

        int _f_index;
        int _hl_index;

        double _H_fuel;
        double _H_ox;

        MixingRxnModel* const _model;

    };

    class SingleMF : public TransformBase {

      //Single mixture fraction assumes that the IVs are ordered:
      //(1) mixture fraction
      //(2) scalar variance
      //(3) heat loss

      public:
        SingleMF( std::map<std::string, double>& keys, MixingRxnModel* const model);
        ~SingleMF();

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool sf_transform = false;
          ProblemSpecP p = ps;

          if ( p->findBlock("standard_flamelet" )){

            ProblemSpecP p_sf = p->findBlock("standard_flamelet");
            p_sf->getAttribute("f_label",_f_name);
            p_sf->getAttribute("var_label", _var_name);
            p_sf->getAttribute("hl_label", _hl_name);
            sf_transform = true;

          } else if ( p->findBlock("standard_equilibrium") ){

            ProblemSpecP p_sf = p->findBlock("standard_equilibrium");
            p_sf->getAttribute("f_label",_f_name);
            p_sf->getAttribute("var_label", _var_name);
            p_sf->getAttribute("hl_label", _hl_name);
            sf_transform = true;

          }

          if ( p->findBlock("reference_state") ){
            p->findBlock("reference_state")->getAttribute("f",_f_ref);
            p->findBlock("reference_state")->getAttribute("hl",_hl_ref);
            p->findBlock("reference_state")->getAttribute("var",_var_ref);
          } else {
            throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
          }

          if ( sf_transform ){
            _f_index= -1;
            _hl_index = -1;
            _var_index = -1;

            int index = 0;
            for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

              if ( *i == _f_name){
                _f_index = index;
              } else if ( *i == _var_name ){
                _var_index = index;
              } else if ( *i == _hl_name ){
                _hl_index = index;
              }
              index++;

            }
            if ( _f_index == -1 ) {
              proc0cout << "Warning: Could not match mixture fraction label to table variables!" << std::endl;
              sf_transform = false;
            }
            if ( _var_index == -1 ) {
              proc0cout << "Warning: Could not match variance label to table variables!" << std::endl;
              sf_transform = false;
            }
            if ( _hl_index == -1 ) {
              proc0cout << "Warning: Could not match heat loss label to table variables!" << std::endl;
              sf_transform = false;
            }
          }

          std::vector<double> my_ivs;
          my_ivs.push_back(1);
          my_ivs.push_back(0);
          my_ivs.push_back(0);
          double h = _model->getTableValue( my_ivs, "adiabaticenthalpy" );
          _H_fuel = h;

          my_ivs[0] = 0;
          h = _model->getTableValue( my_ivs, "adiabaticenthalpy" );
          _H_ox = h;

          return sf_transform;

        };

        bool inline has_heat_loss(){ return true; };

        void inline transform( std::vector<double>& iv, double inert ){

          double f = iv[_f_index];
          double var = iv[_var_index];
          double hl = iv[_hl_index];

          iv[0] = f;
          iv[1] = var;
          iv[2] = hl;

        };

        double inline get_adiabatic_enthalpy( std::vector<double>& iv, double inerts ){
          return iv[0]*_H_fuel + ( 1.0 - iv[0] )*_H_ox;
        };

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        inline const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const  size )
        {
          std::vector<double>  hl_bounds(2);
          hl_bounds[0]=(iv_grids(1, 0));
          hl_bounds[1]=(iv_grids(1, size(2)-1));

          return hl_bounds;
        };
#endif

        const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> const size )
        {
          std::vector<double>  hl_bounds(2);
          hl_bounds[0]=(iv_grids[1][0]);
          hl_bounds[1]=(iv_grids[1][size[2]-1]);

          return hl_bounds;
        };

        struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv(){
          struct1DArray<double,ALMOST_A_MAGIC_NUMBER> iv(3);
          iv[0]=_f_ref;
          iv[1]=_var_ref;
          iv[2]=_hl_ref;

          return iv;
        };

      private:

        std::string _f_name;
        std::string _var_name;
        std::string _hl_name;

        double _f_ref;
        double _var_ref;
        double _hl_ref;

        std::map<std::string,double> _keys;

        int _f_index;
        int _var_index;
        int _hl_index;

        double _H_fuel;
        double _H_ox;

        MixingRxnModel* const _model;

    };

    class CoalTransform : public TransformBase {

      public:
        CoalTransform( std::map<std::string,double>& keys, MixingRxnModel* const model );
        ~CoalTransform();

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool coal_table_on = false;
          ProblemSpecP p = ps;
          bool doit = false;
          _is_acidbase = false;

          std::map<std::string,double>::iterator iter = _keys.find( "transform_constant" );
          if ( iter == _keys.end() ){
            _constant = 0.0;
          } else {
            _constant = iter->second;
          }

          if ( p->findBlock("coal") ){

            p->findBlock("coal")->getAttribute("fp_label", _fp_name );
            p->findBlock("coal")->getAttribute("eta_label", _eta_name );
            p->findBlock("coal")->getAttribute("hl_label",_hl_name);
            doit = true;

            if ( p->findBlock("reference_state") ){
              p->findBlock("reference_state")->getAttribute("fp",_fp_ref);
              p->findBlock("reference_state")->getAttribute("eta",_eta_ref);
              p->findBlock("reference_state")->getAttribute("hl",_hl_ref);
            } else {
              throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
            }

          } else if ( p->findBlock("rcce") ){

            p->findBlock("rcce")->getAttribute("fp_label", _fp_name );
            p->findBlock("rcce")->getAttribute("eta_label", _eta_name );
            p->findBlock("rcce")->getAttribute("hl_label",_hl_name);
            doit = true;

            if ( p->findBlock("reference_state") ){
              p->findBlock("reference_state")->getAttribute("fp",_fp_ref);
              p->findBlock("reference_state")->getAttribute("eta",_eta_ref);
              p->findBlock("reference_state")->getAttribute("hl",_hl_ref);
            } else {
              throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
            }

          } else if ( p->findBlock("acidbase") ){

            p->findBlock("acidbase")->getAttribute("extent_label", _fp_name );
            p->findBlock("acidbase")->getAttribute("f_label",      _eta_name );
            doit = true;
            _is_acidbase = true;

            if ( p->findBlock("reference_state") ){
              p->findBlock("reference_state")->getAttribute("f",_eta_ref);
              p->findBlock("reference_state")->getAttribute("extent",_fp_ref);
            } else {
              throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
            }

          }

          if ( doit ) {

            _eta_index = -1;
            _fp_index = -1;
            _hl_index = -1;

            int index = 0;
            for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

              if ( *i == _eta_name ){
                _eta_index = index;
              } else if ( *i == _fp_name ){
                _fp_index = index;
              } else if ( *i == _hl_name ){
                _hl_index = index;
              }
              index++;

            }
            coal_table_on = true;
            if ( _fp_index == -1 ) {
              proc0cout << "Warning: Could not match PRIMARY mixture fraction label to table variables!" << std::endl;
              coal_table_on = false;
            }
            if ( _eta_index == -1 ) {
              proc0cout << "Warning: Could not match ETA mixture fraction label to table variables!" << std::endl;
              coal_table_on = false;
            }
            if ( !_is_acidbase ){
              if ( _hl_index == -1 ) {
                proc0cout << "Warning: Could not match heat loss label to table variables!" << std::endl;
                coal_table_on = false;
              }
            }

            if ( !_is_acidbase ){

              std::vector<double> my_ivs;
              my_ivs.push_back(0);
              my_ivs.push_back(0);
              my_ivs.push_back(1);
              _H_F1 = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

              my_ivs[2] = 0;
              _H_F0 = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

              my_ivs[0] = 1;
              _H_fuel = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

            }
          }
          return coal_table_on;
        };

        bool inline has_heat_loss(){ return true; };

        void inline transform( std::vector<double>& iv, double inert ){

          double f = 0.0;
          double fp = iv[_fp_index];
          double hl = 0;
          if ( !_is_acidbase )
            hl = iv[_hl_index];
          double eta = iv[_eta_index];

          if ( eta < 1.0 ){

            f = ( fp - _constant * eta ) / ( 1.0 - eta );

            if ( f < 0.0 )
              f = 0.0;
            if ( f > 1.0 )
              f = 1.0;
          }

          iv[0] = eta;
          iv[1] = hl;
          iv[2] = f;

        };

        double inline get_adiabatic_enthalpy( std::vector<double>& iv, double inert ){

          this->transform(iv, 0.0);

          double H2 = iv[2]*_H_F1 + ( 1.0 - iv[2] )*_H_F0;

          double H_ad = iv[0]*_H_fuel + ( 1.0 - iv[0] )*H2;

          return H_ad;

        };

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        inline const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const  size )
        {
          std::vector<double>  hl_bounds(2);
          hl_bounds[0]=(iv_grids(0, 0));
          hl_bounds[1]=(iv_grids(0, size(1)-1));

          return hl_bounds;
        };
#endif

        const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> const size )
        {
          std::vector<double> hl_bounds;
          hl_bounds.push_back(iv_grids[0][0]);
          hl_bounds.push_back(iv_grids[0][size[1]-1]);

          return hl_bounds;
        };

        struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv(){

          if ( _is_acidbase ){
            std::vector<double> iv(2);

            iv[_fp_index] = _fp_ref;
            iv[_eta_index] = _eta_ref;

            this->transform(iv, 0.0);

            struct1DArray<double,ALMOST_A_MAGIC_NUMBER> iv_p(iv,iv.size());  //portable version

            std::cout << "IV=" << iv_p[0] << " " << iv_p[1] << std::endl;

            return iv_p;
          } else {
            std::vector<double> iv(3);

            iv[_fp_index] = _fp_ref;
            iv[_eta_index] = _eta_ref;
            iv[_hl_index] = _hl_ref;

            this->transform(iv, 0.0);

            struct1DArray<double,ALMOST_A_MAGIC_NUMBER> iv_p(iv,iv.size());  //portable version

            return iv_p;
          }
        };


      private:

        double _constant;

        std::string _eta_name;
        std::string _fp_name;
        std::string _hl_name;

        double _fp_ref;
        double _eta_ref;
        double _hl_ref;

        int _eta_index;
        int _fp_index;
        int _hl_index;

        std::map<std::string,double> _keys;

        bool _is_acidbase;

        double _H_F1;
        double _H_F0;
        double _H_fuel;

        MixingRxnModel* const _model;
    };

    class AcidBase: public TransformBase {

      public:
        AcidBase( std::map<std::string,double>& keys, MixingRxnModel* const model );
        ~AcidBase();

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool acid_base_table_on = false;
          ProblemSpecP p = ps;

          bool doit = false;

          std::map<std::string,double>::iterator iter = _keys.find( "transform_constant" );
          if ( iter == _keys.end() ){
            throw InvalidValue("Error: The key: transform_constant is required for this table and wasn't found.",__FILE__,__LINE__);
          } else {
            _constant = iter->second;
          }

          if ( p->findBlock("acidbase") ){

            p->findBlock("acidbase")->getAttribute("extent_label", _fp_name );
            p->findBlock("acidbase")->getAttribute("f_label",      _eta_name );

            if ( p->findBlock("reference_state") ){
              p->findBlock("reference_state")->getAttribute("extent",_fp_ref);
              p->findBlock("reference_state")->getAttribute("f",_eta_ref);
            } else {
              throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
            }

            doit = true;
          }

          if ( doit ) {

            _eta_index = -1;
            _fp_index = -1;

            int index = 0;
            for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

              if ( *i == _eta_name ){
                _eta_index = index;
              } else if ( *i == _fp_name ){
                _fp_index = index;
              }
              index++;

            }

            acid_base_table_on = true;

            if ( _fp_index == -1 ) {
              proc0cout << "Warning: Could not match PRIMARY mixture fraction label to table variables!" << std::endl;
              acid_base_table_on = false;
            }
            if ( _eta_index == -1 ) {
              proc0cout << "Warning: Could not match ETA mixture fraction label to table variables!" << std::endl;
              acid_base_table_on = false;
            }
          }

          return acid_base_table_on;

        };

        bool inline has_heat_loss(){ return false; };

        void inline transform( std::vector<double>& iv, double inert ){

          double f = 0.0;
          double fp = iv[_fp_index];
          double eta = iv[_eta_index];

          if ( eta < 1.0 ){

            f = ( fp - _constant * eta ) / ( 1.0 - eta );

            if ( f < 1.0e-16 )
              f = 0.0;
            if ( f > 1.0 )
              f = 1.0;
          }

          iv[0] = eta;
          iv[1] = f;

        };

        double inline get_adiabatic_enthalpy( std::vector<double>& iv, double inert ){

          throw InvalidValue("Error: No ability to return adiabatic enthalpy for the acid base transform",__FILE__,__LINE__);

        };

        const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> const size )
        {
          throw InvalidValue("Error: No ability to return heat loss bounds for the acid base transform",__FILE__,__LINE__);
        };

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<Kokkos::RandomAccess> >  const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> >  const  size ){
          throw InvalidValue("Error: No ability to return heat loss bounds for the acid base transform",__FILE__,__LINE__);
        };
#endif

        struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv(){
          std::vector<double> iv(2);

          iv[_fp_index] = _fp_ref;
          iv[_eta_index] = _eta_ref;

          this->transform(iv, 0.0);

          struct1DArray<double,ALMOST_A_MAGIC_NUMBER> iv_p(iv,iv.size());  //portable version
          return iv_p;
        };


      private:

        double _constant;

        std::string _eta_name;
        std::string _fp_name;

        double _eta_ref;
        double _fp_ref;

        int _eta_index;
        int _fp_index;

        std::map<std::string,double> _keys;

        MixingRxnModel* const _model;
    };

    class RCCETransform : public TransformBase {

      public:
        RCCETransform( std::map<std::string, double>& keys, MixingRxnModel* const model );
        ~RCCETransform();

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool rcce_table_on = false;
          ProblemSpecP p = ps;

          _rcce_fp  = false;
          _rcce_eta = false;

          if ( p->findBlock("rcce_fp") ){

            p->findBlock("rcce_fp")->getAttribute("fp_label", _fp_name );
            p->findBlock("rcce_fp")->getAttribute("xi_label", _xi_name );
            p->findBlock("rcce_fp")->getAttribute("hl_label", _hl_name );
            _rcce_fp = true;

            if ( p->findBlock("reference_state") ){
              p->findBlock("reference_state")->getAttribute("fp",_fp_ref);
              p->findBlock("reference_state")->getAttribute("xi",_xi_ref);
              p->findBlock("reference_state")->getAttribute("hl",_hl_ref);
            } else {
              throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
            }

          } else if ( p->findBlock("rcce_eta") ){

            p->findBlock("rcce_eta")->getAttribute("eta_label", _eta_name );
            p->findBlock("rcce_eta")->getAttribute("xi_label",  _xi_name );
            p->findBlock("rcce_eta")->getAttribute("hl_label",  _hl_name );
            _rcce_eta = true;

            if ( p->findBlock("reference_state") ){
              p->findBlock("reference_state")->getAttribute("eta",_eta_ref);
              p->findBlock("reference_state")->getAttribute("xi",_xi_ref);
              p->findBlock("reference_state")->getAttribute("hl",_hl_ref);
            } else {
              throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
            }

          }

          if ( _rcce_fp ) {

            _xi_index = -1;
            _fp_index = -1;
            _hl_index = -1;

            int index = 0;
            for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

              if ( *i == _fp_name )
                _fp_index = index;

              if ( *i == _xi_name )
                _xi_index = index;

              if ( *i == _hl_name )
                _hl_index = index;

              index++;

            }

            rcce_table_on = true;

            if ( _fp_index == -1 ) {
              proc0cout << "Warning: Could not match Fp mixture fraction label to table variables!" << std::endl;
              rcce_table_on = false;
            }
            if ( _xi_index == -1 ) {
              proc0cout << "Warning: Could not match Xi mixture fraction label to table variables!" << std::endl;
              rcce_table_on = false;
            }
            if ( _hl_index == -1 ) {
              proc0cout << "Warning: Could not match heat loss label to table variables!" << std::endl;
              rcce_table_on = false;
            }

          } else if ( _rcce_eta ) {

            _xi_index = -1;
            _eta_index = -1;

            int index = 0;
            for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

              if ( *i == _eta_name )
                _eta_index = index;

              if ( *i == _xi_name )
                _xi_index = index;

              if ( *i == _hl_name )
                _hl_index = index;

              index++;

            }

            std::vector<double> my_ivs;
            my_ivs.push_back(0);
            my_ivs.push_back(0);
            my_ivs.push_back(1);
            _H_F1 = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

            my_ivs[2] = 0;
            _H_F0 = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

            my_ivs[0] = 1;
            _H_fuel = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

            rcce_table_on = true;

            if ( _eta_index == -1 ) {
              proc0cout << "Warning: Could not match Eta mixture fraction label to table variables!" << std::endl;
              rcce_table_on = false;
            }
            if ( _xi_index == -1 ) {
              proc0cout << "Warning: Could not match Xi mixture fraction label to table variables!" << std::endl;
              rcce_table_on = false;
            }
            if ( _hl_index == -1 ) {
              proc0cout << "Warning: Could not match heat loss label to table variables!" << std::endl;
              rcce_table_on = false;
            }

          }

          return rcce_table_on;

        };

        bool inline has_heat_loss(){ return true; };

        void inline transform( std::vector<double>& iv, double inert ){

          double f   = 0.0;
          double eta = 0.0;
          double fp  = 0.0;
          double hl  = 0.0;
          double xi  = 0.0;

          if ( _rcce_fp ) {

            fp = iv[_fp_index];
            xi = iv[_xi_index];
            hl = iv[_hl_index];

            eta = xi - fp;

            if ( eta < 1.0 ){
              f = ( fp ) / ( 1.0 - eta );
            } else {
              f = 0.0;
            }

            if ( f < 0.0 )
              f = 0.0;
            if ( f > 1.0 )
              f = 1.0;


          } else if ( _rcce_eta ){

            eta = iv[_eta_index];
            xi  = iv[_xi_index];
            hl  = iv[_hl_index];

            double fp = xi - eta;

            if ( eta < 1.0 ){
              f = ( fp ) / ( 1.0 - eta );
            } else {
              f = 0.0;
            }

            if ( f < 0.0 )
              f = 0.0;
            if ( f > 1.0 )
              f = 1.0;

          }

          //reassign
          iv[0] = eta;
          iv[1]  = hl;
          iv[2]   = f;

        };

        double inline get_adiabatic_enthalpy( std::vector<double>& iv, double inert ){

          this->transform(iv, 0.0);

          double H2 = iv[2]*_H_F1 + ( 1.0 - iv[2] )*_H_F0;

          double H_ad = iv[0]*_H_fuel + ( 1.0 - iv[0] )*H2;

          return H_ad;

        };

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        inline const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const  size )
        {
          std::vector<double>  hl_bounds(2);
          hl_bounds[0]=(iv_grids(0, 0));
          hl_bounds[1]=(iv_grids(0, size(1)-1));

          return hl_bounds;
        };
#endif

        const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> const size )
        {
          std::vector<double> hl_bounds;
          hl_bounds.push_back(iv_grids[0][0]);
          hl_bounds.push_back(iv_grids[0][size[1]-1]);

          return hl_bounds;
        };

        struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv(){
          std::vector<double> iv(3);

          if ( _rcce_fp ){
            iv[_fp_index] = _fp_ref;
            iv[_xi_index] = _xi_ref;
            iv[_hl_index] = _hl_ref;
          } else if ( _rcce_eta ){
            iv[_eta_index] = _eta_ref;
            iv[_xi_index] = _xi_ref;
            iv[_hl_index] = _hl_ref;
          }

          this->transform(iv, 0.0);

          struct1DArray<double,ALMOST_A_MAGIC_NUMBER> iv_p(3);
          return iv_p;
        };

      private:

        bool _rcce_eta;
        bool _rcce_fp;

        std::string _eta_name;
        std::string _hl_name;
        std::string _fp_name;
        std::string _xi_name;

        double _eta_ref;
        double _fp_ref;
        double _xi_ref;
        double _hl_ref;

        int _eta_index;
        int _fp_index;
        int _hl_index;
        int _xi_index;

        int _table_eta_index;
        int _table_hl_index;
        int _table_f_index;

        double _H_F1;
        double _H_F0;
        double _H_fuel;

        std::map<std::string, double> _keys;

        MixingRxnModel* const _model;

    };

    class InertMixing : public TransformBase {

      public:
        InertMixing( std::map<std::string, double>& keys, MixingRxnModel* const model );
        ~InertMixing();

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){
          bool transform_on = false;
          ProblemSpecP p = ps;
          bool doit = false;
          if ( p->findBlock("inert_mixing") ){

            p->findBlock("inert_mixing")->getAttribute("fp_label",  _fp_name );
            p->findBlock("inert_mixing")->getAttribute("eta_label", _eta_name );
            p->findBlock("inert_mixing")->getAttribute("hl_label",  _hl_name );

            if ( p->findBlock("reference_state") ){
              p->findBlock("reference_state")->getAttribute("eta",_eta_ref);
              p->findBlock("reference_state")->getAttribute("fp",_fp_ref);
              p->findBlock("reference_state")->getAttribute("hl",_hl_ref);
            } else {
              throw ProblemSetupException("Error: Reference state not defined.",__FILE__, __LINE__ );
            }

            doit = true;

          }

          if ( doit ) {

            _eta_index = -1;
            _fp_index  = -1;
            _hl_index  = -1;

            int index = 0;
            for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

              if ( *i == _fp_name ){
                _fp_index = index;
              } else if ( *i == _eta_name ){
                _eta_index = index;
              } else if ( *i == _hl_name ){
                _hl_index = index;
              }

              index++;

            }
            transform_on = true;
            if ( _eta_index == -1 ) {
              proc0cout << "Warning: Could not match Eta mixture fraction label to table variables!" << std::endl;
              transform_on = false;
            }
            if ( _fp_index == -1 ) {
              proc0cout << "Warning: Could not match Fp mixture fraction label to table variables!" << std::endl;
              transform_on = false;
            }
            if ( _hl_index == -1 ){
              proc0cout << "Warning: Could not match heat loss label to table variables!" << std::endl;
              transform_on = false;
            }
          }

          std::vector<double> my_ivs;
          my_ivs.push_back(0);
          my_ivs.push_back(0);
          my_ivs.push_back(1);
          _H_F1 = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

          my_ivs[2] = 0;
          _H_F0 = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

          my_ivs[0] = 1;
          _H_fuel = _model->getTableValue( my_ivs, "adiabaticenthalpy" );

          return transform_on;

        };

        void inline transform( std::vector<double>& iv, double inert ){

          double fcstar = iv[_fp_index];
          double fc = iv[_eta_index];
          double hl = iv[_hl_index];

          if ( inert < 1.0 ){

            double eta = fc / ( 1.0 - inert );
            double fp  = fcstar / ( 1.0 - inert );

            if ( fp > 1.0 ) fp = 1.0;

            double f = 0.0;

            if ( eta < 1.0 ){

              f   = fp / ( 1.0 - eta );

              if ( f < 0.0 )
                f = 0.0;
              if ( f > 1.0 )
                f = 1.0;
            }

            iv[0] = eta;
            iv[1] = hl;
            iv[2] = f;

          } else {

            iv[0] = 0.0;
            iv[1] = hl;
            iv[2] = 0.0;

          }

        };

        double inline get_adiabatic_enthalpy( std::vector<double>& iv, double inert ){

          this->transform(iv, inert);

          double H2 = iv[2]*_H_F1 + ( 1.0 - iv[2] )*_H_F0;

          double H_ad = iv[0]*_H_fuel + ( 1.0 - iv[0] )*H2;

          return H_ad;

        };

        bool inline has_heat_loss(){ return true; };

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
        inline const std::vector<double> get_hl_bounds( Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const iv_grids,Kokkos::View<int*,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > const  size )
        {
          std::vector<double>  hl_bounds(2);
          hl_bounds[0]=(iv_grids(0, 0));
          hl_bounds[1]=(iv_grids(0, size(1)-1));

          return hl_bounds;
        };
#endif

        const std::vector<double> get_hl_bounds( std::vector<std::vector<double> > const iv_grids, std::vector<int> const size )
        {
          std::vector<double> hl_bounds;
          hl_bounds.push_back(iv_grids[0][0]);
          hl_bounds.push_back(iv_grids[0][size[1]-1]);

          return hl_bounds;
        };

        struct1DArray<double,ALMOST_A_MAGIC_NUMBER> get_reference_iv(){
          std::vector<double> iv(3);

          iv[_fp_index] = _fp_ref;
          iv[_eta_index] = _eta_ref;
          iv[_hl_index] = _hl_ref;

          this->transform(iv, 0.0);

          struct1DArray<double,ALMOST_A_MAGIC_NUMBER> iv_p(3);
          return iv_p;
        };

      private:

        std::string _eta_name;
        std::string _fp_name;
        std::string _hl_name;

        int _eta_index;
        int _fp_index;
        int _hl_index;

        double _eta_ref;
        double _fp_ref;
        double _hl_ref;

        double _H_F1;
        double _H_F0;
        double _H_fuel;

        std::map<std::string, double> _keys;
        MixingRxnModel* const _model;

    };


  public:

    TransformBase* _iv_transform;

    /** @brief Check to ensure all BCs are set **/
    void sched_checkTableBCs( const LevelP& level, SchedulerP& sched );
    void checkTableBCs( const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw );


  protected:

    /** @brief Performs post mixing on table look up value based
     * on a set of inert streams set from the input file */
    void post_mixing( double& mixvalue, double f, std::string label, doubleMap& the_map ){

      // mixvalue is coming in with the post table-lookup value.

      doubleMap::iterator I = the_map.find( label );
      double i_value = I->second;

      if ( I != the_map.end() ){

        mixvalue = i_value * f + mixvalue * ( 1.0 - f );

      } else {

        // can't find it in the list.  Assume it is zero.
        mixvalue = ( 1.0 - f ) * mixvalue;

      }

    };

    /** @brief Performs post mixing on table look up value based
     * on a set of inert streams set from the input file.  Fails if
     * the variable isn't found */
    void strict_post_mixing( double& mixvalue, double f, std::string label, doubleMap& the_map ){

      // mixvalue is coming in with the post table-lookup value.

      doubleMap::iterator I = the_map.find( label );
      double i_value = I->second;

      if ( I != the_map.end() ){

        mixvalue = i_value * f + mixvalue * ( 1.0 - f );

      } else {

        // can't find it in the list.  Throw an error:
        throw InvalidValue("Error: Attempting to post-mix in "+label+" but variable is not found in this inert list.  Check your input file.",__FILE__,__LINE__);

      }

    };

    VarMap d_dvVarMap;         ///< Dependent variable map
    VarMap d_oldDvVarMap;      ///< Dependent variable map from previous lookup
    VarMap d_ivVarMap;         ///< Independent variable map
    doubleMap d_constants;     ///< List of constants in table header
    InertMasterMap d_inertMap; ///< List of inert streams for post table lookup mixing

    const VarLabel* m_timeStepLabel;

    const VarLabel* m_denRefArrayLabel;
    const VarLabel* m_densityLabel;
    const VarLabel* m_volFractionLabel;

    /** @brief Sets the mixing table's dependent variable list. */
    void setMixDVMap( const ProblemSpecP& root_params );

    /** @brief Common problem setup work */
    void problemSetupCommon( const ProblemSpecP& params, MixingRxnModel* const model );

    MaterialManagerP& m_materialManager;    ///< Material Manager
    int m_matl_index;                       ///< Arches material index

    bool d_coldflow;                        ///< Will not compute heat loss and will not initialized ethalpy
    bool d_adiabatic;                       ///< Will not compute heat loss
    bool d_does_post_mixing;                ///< Turn on/off post mixing of inerts
    bool d_has_transform;                   ///< Indicates if a variable transform is used
    bool d_user_ref_density;                ///< Indicates if the user is setting the reference density from input

    double d_reference_density;             ///< User defined reference density

    std::string d_fp_label;                 ///< Primary mixture fraction name for a coal table
    std::string d_eta_label;                ///< Eta mixture fraction name for a coal table
    std::vector<std::string> d_allIndepVarNames; ///< Vector storing all independent variable names from table file
    std::vector<std::string> d_allDepVarNames;   ///< Vector storing all dependent variable names from the table file

    /** @brief Insert a varLabel into the map where the varlabel has been created elsewhere */
    inline void insertExisitingLabelIntoMap( const std::string var_name ){
      VarMap::iterator i = d_dvVarMap.find( var_name );
      if ( i == d_dvVarMap.end() ) {
        const VarLabel* the_label = VarLabel::find(var_name);
        i = d_dvVarMap.insert( make_pair( var_name, the_label ) ).first;
        proc0cout << " creating a label for  ---> " << var_name << std::endl;
      }
      return;
    }
  }; // end class MixingRxnModel
} // end namespace Uintah

#endif
