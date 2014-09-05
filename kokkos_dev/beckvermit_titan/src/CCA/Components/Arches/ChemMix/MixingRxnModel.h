/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Mixing/Stream.h>

// Uintah includes
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DebugStream.h>

// C++ includes
#include     <vector>
#include     <map>
#include     <string>
#include     <stdexcept>


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

  class ArchesLabel; 
  class TimeIntegratorLabel; 
  class MixingRxnModel{

  public:

    // Useful typedefs
    typedef std::map<string, const VarLabel* >           VarMap;
    typedef std::map<string, CCVariable<double>* >       CCMap; 
    typedef std::map<string, double >                    doubleMap; 
    typedef std::map<string, doubleMap>                  InertMasterMap; 
    struct ConstVarContainer {

      constCCVariable<double> var; 

    }; 
    typedef std::map<std::string, ConstVarContainer> StringToCCVar; 

    MixingRxnModel( ArchesLabel* labels, const MPMArchesLabel* MAlabels );

    virtual ~MixingRxnModel();

    /** @brief Interface the the input file.  Get table name, then read table data into object */
    virtual void problemSetup( const ProblemSpecP& params ) = 0;

    /** @brief Returns a vector of the state space for a given set of independent parameters */
    virtual void sched_getState( const LevelP& level, 
        SchedulerP& sched, 
        const TimeIntegratorLabel* time_labels, 
        const bool initialize,
        const bool with_energy_exch,
        const bool modify_ref_den ) = 0;

    /** @brief Provides access for models, algorithms, etc. to add additional table lookup variables. */
    void addAdditionalDV( std::vector<string>& vars );

    /** @brief Needed for the old properties method until it goes away */ 
    virtual void oldTableHack( const InletStream&, Stream&, bool, const std::string ) = 0; 

    /** @brief Needed for dumb MPMArches */ 
    virtual void sched_dummyInit( const LevelP& level, 
        SchedulerP& sched ) = 0;

    /** @brief Returns the value of a single variable given the iv vector 
     * This will be useful for other classes to have access to */
    virtual double getTableValue( std::vector<double>, std::string ) = 0; 

		/** @brief Get a table value **/ 
    virtual double getTableValue( std::vector<double> iv, std::string depend_varname, 
        StringToCCVar inert_mixture_fractions, IntVector c) = 0;

    /** @brief Get a table value **/ 
    virtual double getTableValue( std::vector<double> iv, std::string depend_varname, 
                   doubleMap inert_mixture_fractions ) = 0;
		
		/** @brief For efficiency: Matches tables lookup species with pointers/index/etc */
		virtual void tableMatching() = 0; 

    /** @brief Return a reference to the independent variables */
    inline const VarMap getIVVars(){ return d_ivVarMap; }; 

    /** @brief Return a reference to the dependent variables */ 
    inline const VarMap getDVVars(){ return d_dvVarMap; }; 

    /** @brief Return a string list of all independent variable names in order */ 
    inline std::vector<string>& getAllIndepVars(){ return d_allIndepVarNames; }; 

    /** @brief Return a string list of dependent variables names in the order they were read */ 
    inline std::vector<string>& getAllDepVars(){ return d_allDepVarNames; };
  
    /** @brief Returns a <string, double> map of KEY constants found in the table */ 
    inline doubleMap& getAllConstants(){ return d_constants; };

    /** @brief Returns the map of participating inerts **/
    inline InertMasterMap& getInertMap(){ return d_inertMap; }; 

    /** @brief Returns a boolean regarding if post mixing is used or not **/ 
    inline bool doesPostMix(){ return d_does_post_mixing; }; 
  
    /** @brief  Insert the name of a dependent variable into the dependent variable map (dvVarMap), which maps strings to VarLabels */
    inline void insertIntoMap( const string var_name ){

      VarMap::iterator i = d_dvVarMap.find( var_name ); 

      if ( i == d_dvVarMap.end() ) {

        const VarLabel* the_label = VarLabel::create( var_name, CCVariable<double>::getTypeDescription() ); 

        d_dvVarMap.insert( std::make_pair( var_name, the_label ) ); 

        proc0cout << "    ---> " << var_name << endl; 

      } 
      return; 
    };

    inline double get_Ha( std::vector<double>& iv, double inerts ){ 

      double ha = _iv_transform->get_adiabatic_enthalpy( iv, inerts ); 
      return ha; 

    } 

    /** @brief Get a dependant variable's index **/ 
    virtual int findIndex(std::string) = 0; 

    /** @brief Returns false if flow is changing temperature */ 
    // In other words, is temperature changing? 
    // The strange logic is to make the logic less confusing in Properties.cc
    bool is_not_cold(){ if (d_coldflow){ return false; }else{ return true; }};
      
  protected :

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
        virtual const vector<double> get_hl_bounds( vector<vector<double> > const iv_grids, vector<int> size ) = 0;  

        /** @brief Check to see if this table deals with heat loss **/ 
        virtual bool has_heat_loss() = 0; 

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

        const vector<double> get_hl_bounds( vector<vector<double> > const iv_grids, vector<int> size )
        {
          vector<double> hl_bounds; 
          hl_bounds.push_back(-1.0);
          hl_bounds.push_back(1.0);
          return hl_bounds; 
        };  
    };

    class SingleMF : public TransformBase { 

      //Single mixture fraction assumes that the IVs are ordered: 
      //(1) mixture fraction
      //(2) scalar variance
      //(3) heat loss

      public: 
        SingleMF( std::map<string, double>& keys, MixingRxnModel* const model);
        ~SingleMF(); 

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool sf_transform = false; 
          ProblemSpecP p = ps; 
          typedef std::map<std::string,double> key_map;

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
              proc0cout << "Warning: Could not match mixture fraction label to table variables!" << endl;
              sf_transform = false; 
            }
            if ( _var_index == -1 ) {
              proc0cout << "Warning: Could not match variance label to table variables!" << endl;
              sf_transform = false; 
            }
            if ( _hl_index == -1 ) {
              proc0cout << "Warning: Could not match heat loss label to table variables!" << endl;
              sf_transform = false; 
            }
          } 

          vector<double> my_ivs;
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

        const vector<double> get_hl_bounds( vector<vector<double> > const iv_grids, vector<int> const size )
        {
          vector<double> hl_bounds; 

          hl_bounds.push_back(iv_grids[1][0]);
          hl_bounds.push_back(iv_grids[1][size[2]-1]);

          return hl_bounds; 
        };  

      private: 

        string _f_name; 
        string _var_name; 
        string _hl_name; 

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
        CoalTransform( std::map<string,double>& keys, MixingRxnModel* const model ); 
        ~CoalTransform(); 

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool coal_table_on = false; 
          ProblemSpecP p = ps; 
          bool doit = false; 
          _is_acidbase = false; 
          typedef std::map<std::string,double> key_map;

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

          } else if ( p->findBlock("rcce") ){ 

            p->findBlock("rcce")->getAttribute("fp_label", _fp_name );
            p->findBlock("rcce")->getAttribute("eta_label", _eta_name ); 
            p->findBlock("rcce")->getAttribute("hl_label",_hl_name); 
            doit = true; 

          } else if ( p->findBlock("acidbase") ){

            p->findBlock("acidbase")->getAttribute("extent_label", _fp_name );
            p->findBlock("acidbase")->getAttribute("f_label",      _eta_name ); 
            doit = true; 
            _is_acidbase = true; 

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
              proc0cout << "Warning: Could not match PRIMARY mixture fraction label to table variables!" << endl;
              coal_table_on = false; 
            }
            if ( _eta_index == -1 ) {
              proc0cout << "Warning: Could not match ETA mixture fraction label to table variables!" << endl;
              coal_table_on = false; 
            }
            if ( !_is_acidbase ){ 
              if ( _hl_index == -1 ) {
                proc0cout << "Warning: Could not match heat loss label to table variables!" << endl;
                coal_table_on = false; 
              }
            } 

            if ( !_is_acidbase ){ 

              vector<double> my_ivs;
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

        const vector<double> get_hl_bounds( vector<vector<double> > const iv_grids, vector<int> const size )
        {
          vector<double> hl_bounds; 
          hl_bounds.push_back(iv_grids[0][0]);
          hl_bounds.push_back(iv_grids[0][size[1]-1]);

          return hl_bounds; 
        };  


      private: 

        double _constant; 

        std::string _eta_name; 
        std::string _fp_name; 
        std::string _hl_name; 

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
        AcidBase( std::map<string,double>& keys, MixingRxnModel* const model ); 
        ~AcidBase(); 

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool acid_base_table_on = false; 
          ProblemSpecP p = ps; 

          bool doit = false; 
          typedef std::map<std::string,double> key_map;

          std::map<std::string,double>::iterator iter = _keys.find( "transform_constant" ); 
          if ( iter == _keys.end() ){ 
            throw InvalidValue("Error: The key: transform_constant is required for this table and wasn't found.",__FILE__,__LINE__); 
          } else { 
            _constant = iter->second; 
          }

          if ( p->findBlock("acidbase") ){

            p->findBlock("acidbase")->getAttribute("extent_label", _fp_name );
            p->findBlock("acidbase")->getAttribute("f_label",      _eta_name ); 

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
              proc0cout << "Warning: Could not match PRIMARY mixture fraction label to table variables!" << endl;
              acid_base_table_on = false; 
            }
            if ( _eta_index == -1 ) {
              proc0cout << "Warning: Could not match ETA mixture fraction label to table variables!" << endl;
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

            if ( f < 0.0 )
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

        const vector<double> get_hl_bounds( vector<vector<double> > const iv_grids, vector<int> const size )
        {
          throw InvalidValue("Error: No ability to return heat loss bounds for the acid base transform",__FILE__,__LINE__); 
        };  


      private: 

        double _constant; 

        std::string _eta_name; 
        std::string _fp_name; 

        int _eta_index; 
        int _fp_index; 

        std::map<std::string,double> _keys;

        MixingRxnModel* const _model; 
    };

    class RCCETransform : public TransformBase {

      public: 
        RCCETransform( std::map<string, double>& keys, MixingRxnModel* const model ); 
        ~RCCETransform(); 

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){

          bool rcce_table_on = false; 
          ProblemSpecP p = ps; 
          typedef std::map<std::string,double> key_map;

          _rcce_fp  = false; 
          _rcce_eta = false; 

          if ( p->findBlock("rcce_fp") ){

            p->findBlock("rcce_fp")->getAttribute("fp_label", _fp_name );
            p->findBlock("rcce_fp")->getAttribute("xi_label", _xi_name ); 
            p->findBlock("rcce_fp")->getAttribute("hl_label", _hl_name ); 
            _rcce_fp = true; 

          } else if ( p->findBlock("rcce_eta") ){ 

            p->findBlock("rcce_eta")->getAttribute("eta_label", _eta_name );
            p->findBlock("rcce_eta")->getAttribute("xi_label",  _xi_name ); 
            p->findBlock("rcce_eta")->getAttribute("hl_label",  _hl_name ); 
            _rcce_eta = true; 

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
              proc0cout << "Warning: Could not match Fp mixture fraction label to table variables!" << endl;
              rcce_table_on = false; 
            }
            if ( _xi_index == -1 ) {
              proc0cout << "Warning: Could not match Xi mixture fraction label to table variables!" << endl;
              rcce_table_on = false; 
            }
            if ( _hl_index == -1 ) {
              proc0cout << "Warning: Could not match heat loss label to table variables!" << endl;
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

            vector<double> my_ivs;
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
              proc0cout << "Warning: Could not match Eta mixture fraction label to table variables!" << endl;
              rcce_table_on = false; 
            }
            if ( _xi_index == -1 ) {
              proc0cout << "Warning: Could not match Xi mixture fraction label to table variables!" << endl;
              rcce_table_on = false; 
            }
            if ( _hl_index == -1 ) {
              proc0cout << "Warning: Could not match heat loss label to table variables!" << endl;
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

        const vector<double> get_hl_bounds( vector<vector<double> > const iv_grids, vector<int> const size )
        {
          vector<double> hl_bounds; 
          hl_bounds.push_back(iv_grids[0][0]);
          hl_bounds.push_back(iv_grids[0][size[1]-1]);

          return hl_bounds; 
        };  

      private: 

        bool _rcce_eta; 
        bool _rcce_fp; 

        std::string _eta_name;
        std::string _hl_name; 
        std::string _fp_name; 
        std::string _xi_name; 

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
        InertMixing( std::map<string, double>& keys, MixingRxnModel* const model ); 
        ~InertMixing(); 

        bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){
          bool transform_on = false; 
          ProblemSpecP p = ps; 
          bool doit = false; 
          if ( p->findBlock("inert_mixing") ){

            p->findBlock("inert_mixing")->getAttribute("fp_label",  _fp_name );
            p->findBlock("inert_mixing")->getAttribute("eta_label", _eta_name );
            p->findBlock("inert_mixing")->getAttribute("hl_label",  _hl_name );
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
              proc0cout << "Warning: Could not match Eta mixture fraction label to table variables!" << endl;
              transform_on = false; 
            }
            if ( _fp_index == -1 ) {
              proc0cout << "Warning: Could not match Fp mixture fraction label to table variables!" << endl;
              transform_on = false; 
            }
            if ( _hl_index == -1 ){ 
              proc0cout << "Warning: Could not match heat loss label to table variables!" << endl;
              transform_on = false; 
            } 
          } 

          vector<double> my_ivs;
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

        const vector<double> get_hl_bounds( vector<vector<double> > const iv_grids, vector<int> const size )
        {
          vector<double> hl_bounds; 
          hl_bounds.push_back(iv_grids[0][0]);
          hl_bounds.push_back(iv_grids[0][size[1]-1]);

          return hl_bounds; 
        };  

      private:

        std::string _eta_name; 
        std::string _fp_name; 
        std::string _hl_name; 

        int _eta_index; 
        int _fp_index; 
        int _hl_index; 

        double _H_F1; 
        double _H_F0; 
        double _H_fuel;

        std::map<std::string, double> _keys; 
        MixingRxnModel* const _model; 

    };


  public: 

    TransformBase* _iv_transform; 

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
    VarMap d_ivVarMap;         ///< Independent variable map
    doubleMap d_constants;     ///< List of constants in table header
    InertMasterMap d_inertMap; ///< List of inert streams for post table lookup mixing

    /** @brief Sets the mixing table's dependent variable list. */
    void setMixDVMap( const ProblemSpecP& root_params ); 

    /** @brief Common problem setup work */ 
    void problemSetupCommon( const ProblemSpecP& params, MixingRxnModel* const model ); 

    ArchesLabel* d_lab;                     ///< Arches labels
    const MPMArchesLabel* d_MAlab;          ///< MPMArches labels

    bool d_coldflow;                        ///< Will not compute heat loss and will not initialized ethalpy
    bool d_adiabatic;                       ///< Will not compute heat loss
    bool d_does_post_mixing;                ///< Turn on/off post mixing of inerts
    bool d_has_transform;                   ///< Indicates if a variable transform is used

    std::string d_fp_label;                 ///< Primary mixture fraction name for a coal table
    std::string d_eta_label;                ///< Eta mixture fraction name for a coal table
    std::vector<string> d_allIndepVarNames; ///< Vector storing all independent variable names from table file
    std::vector<string> d_allDepVarNames;   ///< Vector storing all dependent variable names from the table file

    /** @brief Insert a varLabel into the map where the varlabel has been created elsewhere */ 
    inline void insertExisitingLabelIntoMap( const string var_name ){ 

      VarMap::iterator i = d_dvVarMap.find( var_name ); 

      if ( i == d_dvVarMap.end() ) {

        const VarLabel* the_label = VarLabel::find(var_name); 

        i = d_dvVarMap.insert( make_pair( var_name, the_label ) ).first; 

        proc0cout << " creating a label for  ---> " << var_name << endl; 

      } 
      return; 

    } 

  }; // end class MixingRxnModel
} // end namespace Uintah

#endif
