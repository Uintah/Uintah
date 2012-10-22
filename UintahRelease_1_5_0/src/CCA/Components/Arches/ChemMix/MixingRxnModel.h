/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
      typedef std::map<string, doubleMap>                   InertMasterMap; 

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

      /** @brief Computes the heat loss value */ 
      virtual void sched_computeHeatLoss( const LevelP& level, 
          SchedulerP& sched,
          const bool initialize, const bool calcEnthalpy ) = 0;

      /** @brief Initializes the enthalpy for the first time step */ 
      virtual void sched_computeFirstEnthalpy( const LevelP& level, 
          SchedulerP& sched ) = 0; 

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

      /** @brief Returns false if flow is changing temperature */ 
      // In other words, is temperature changing? 
      // The strange logic is to make the logic less confusing in Properties.cc
      bool is_not_cold(){ if (d_coldflow){ return false; }else{ return true; }};

      double get_ox_enthalpy(){ return _H_ox; }; 

    protected :

      std::string _temperature_label_name; 

      class TransformBase { 

        public: 
          TransformBase();
          virtual ~TransformBase(); 

          virtual bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ) = 0;  
          virtual void transform( std::vector<double>& iv ) = 0; 

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
          void inline transform( std::vector<double>& iv ){};
      };

      class CoalTransform : public TransformBase {

        public: 
          CoalTransform( double constant ); 
          ~CoalTransform(); 

          bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){
            bool coal_table_on = false; 
            ProblemSpecP p = ps; 
            bool doit = false; 
            if ( p->findBlock("coal") ){

              p->findBlock("coal")->getAttribute("fp_label", _index_1_name );
              p->findBlock("coal")->getAttribute("eta_label", _index_2_name ); 
              doit = true; 

            } else if ( p->findBlock("acidbase") ){

              p->findBlock("acidbase")->getAttribute("fp_label", _index_1_name );
              p->findBlock("acidbase")->getAttribute("eta_label", _index_2_name ); 
              doit = true; 

            } 

            if ( doit ) { 

              _index_1 = -1; 
              _index_2 = -1; 

              int index = 0; 
              for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

                if ( *i == _index_1_name ) 
                  _index_1 = index; 
                if ( *i == _index_2_name )
                  _index_2 = index; 
                index++; 

              }
              coal_table_on = true; 
              if ( _index_1 == -1 ) {
                proc0cout << "Warning: Could not match PRIMARY mixture fraction label to table variables!" << endl;
                coal_table_on = false; 
              }
              if ( _index_2 == -1 ) {
                proc0cout << "Warning: Could not match ETA mixture fraction label to table variables!" << endl;
                coal_table_on = false; 
              }
            } 
            return coal_table_on; 
          };  

          void inline transform( std::vector<double>& iv ){
            double f = 0.0; 
            if ( iv[_index_2] < 1.0 ){

              f = ( iv[_index_1] - d_constant * iv[_index_2] ) / ( 1.0 - iv[_index_2] ); 

              if ( f < 0.0 )
                f = 0.0;
              if ( f > 1.0 )
                f = 1.0; 
            }
            iv[_index_1] = f; 
          
          }; 


        private: 

          double d_constant; 
      };

      class InertMixing : public TransformBase {

        public: 
          InertMixing(); 
          ~InertMixing(); 

          bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){
            bool transform_on = false; 
            ProblemSpecP p = ps; 
            bool doit = false; 
            if ( p->findBlock("inertmixing") ){

              p->findBlock("inertmixing")->getAttribute("eta_label",     _index_1_name );
              p->findBlock("inertmixing")->getAttribute("fp_label",         _index_2_name );
              p->findBlock("inertmixing")->getAttribute("inert_label",     _inert_mf ); 
              doit = true; 

            } 

            if ( doit ) { 

              _index_1 = -1; 
              _index_2 = -1; 

              int index = 0; 
              for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

                if ( *i == _index_1_name ) 
                  _index_1 = index; 
                if ( *i == _index_2_name )
                  _index_2 = index; 
                index++; 

              }
              transform_on = true; 
              if ( _index_1 == -1 ) {
                proc0cout << "Warning: Could not match f_cstar mixture fraction label to table variables!" << endl;
                transform_on = false; 
              }
              if ( _index_2 == -1 ) {
                proc0cout << "Warning: Could not match f_c mixture fraction label to table variables!" << endl;
                transform_on = false; 
              }
            } 
            return transform_on; 
          };  

          void inline transform( std::vector<double>& iv ){
            throw InvalidValue("Error: You have chosen to use post mixing but there is something wrong with your tranform.  Check your input file.",__FILE__,__LINE__); 
          }; 

          void inline transform( std::vector<double>& iv, double inert ){

            double fc = iv[_index_2];
            double fcstar = iv[_index_1]; 

            iv[_index_1] = fcstar / ( 1.0 - inert ); 
            double fp = ( fcstar + fc ) / ( 1.0 - inert );
            iv[_index_2] = (1.0 - fp)/iv[_index_1];  

          };

        private:
          std::string _inert_mf;

      };

      class SlowFastTransform : public TransformBase {

        public: 
          SlowFastTransform(); 
          ~SlowFastTransform(); 

          bool problemSetup( const ProblemSpecP& ps, std::vector<std::string> names ){
            bool transform_on = false; 
            ProblemSpecP p = ps; 
            bool doit = false; 
            if ( p->findBlock("slowfastchem") ){

              p->findBlock("slowfastchem")->getAttribute("fp_label", _index_1_name );
              p->findBlock("slowfastchem")->getAttribute("eta_label", _index_2_name ); 
              doit = true; 

            } 

            if ( doit ) { 

              _index_1 = -1; 
              _index_2 = -1; 

              int index = 0; 
              for ( std::vector<std::string>::iterator i = names.begin(); i != names.end(); i++ ){

                if ( *i == _index_1_name ) 
                  _index_1 = index; 
                if ( *i == _index_2_name )
                  _index_2 = index; 
                index++; 

              }
              transform_on = true; 
              if ( _index_1 == -1 ) {
                proc0cout << "Warning: Could not match PRIMARY mixture fraction label to table variables!" << endl;
                transform_on = false; 
              }
              if ( _index_2 == -1 ) {
                proc0cout << "Warning: Could not match ETA mixture fraction label to table variables!" << endl;
                transform_on = false; 
              }
            } 
            return transform_on; 
          };  

          void inline transform( std::vector<double>& iv ){
            iv[_index_2] = iv[_index_1] + iv[_index_2];
            iv[_index_1] = 0.0; 
          }; 
      };

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
      struct ConstVarContainer {

        constCCVariable<double> var; 

      }; 
      typedef std::map<std::string, ConstVarContainer> StringToCCVar; 

      /** @brief Sets the mixing table's dependent variable list. */
      void setMixDVMap( const ProblemSpecP& root_params ); 

      /** @brief Common problem setup work */ 
      void problemSetupCommon( const ProblemSpecP& params ); 

      ArchesLabel* d_lab;                     ///< Arches labels
      const MPMArchesLabel* d_MAlab;          ///< MPMArches labels
      TransformBase* _iv_transform;           ///< Tool for mapping mixture fractions 

      bool d_coldflow;                        ///< Will not compute heat loss and will not initialized ethalpy
      bool d_adiabatic;                       ///< Will not compute heat loss
      bool d_use_mixing_model;                ///< Turn on/off mixing model
      bool d_does_post_mixing;                ///< Turn on/off post mixing of inerts

      std::string d_fp_label;                 ///< Primary mixture fraction name for a coal table
      std::string d_eta_label;                ///< Eta mixture fraction name for a coal table
      std::vector<string> d_allIndepVarNames; ///< Vector storing all independent variable names from table file
      std::vector<string> d_allDepVarNames;   ///< Vector storing all dependent variable names from the table file

      double _H_ox;                          ///< Adiabatic air enthalpy
      double _H_fuel;                        ///< Adiabatic fuel enthalpy

      /** @brief Insert a varLabel into the map where the varlabel has been created elsewhere */ 
      inline void insertExisitingLabelIntoMap( const string var_name ){ 

        VarMap::iterator i = d_dvVarMap.find( var_name ); 

        if ( i == d_dvVarMap.end() ) {

          const VarLabel* the_label = VarLabel::find(var_name); 

          i = d_dvVarMap.insert( make_pair( var_name, the_label ) ).first; 

          proc0cout << "    ---> " << var_name << endl; 

        } 
        return; 

      } 

  }; // end class MixingRxnModel
} // end namespace Uintah

#endif
