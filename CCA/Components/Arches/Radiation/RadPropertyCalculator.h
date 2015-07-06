#ifndef Uintah_Component_Arches_RadPropertyCalculator_h
#define Uintah_Component_Arches_RadPropertyCalculator_h

#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <vector>
#include <Core/Containers/StaticArray.h>

#include <sci_defs/uintah_defs.h>

#ifdef HAVE_RADPROPS
#  include <radprops/AbsCoeffGas.h>
#  include <radprops/RadiativeSpecies.h>
#  include <radprops/Particles.h>
#endif

namespace Uintah { 

  class RadPropertyCalculator{ 

    public: 

      RadPropertyCalculator( const int _matl_index );

      ~RadPropertyCalculator();

      typedef std::vector<constCCVariable<double> > RadCalcSpeciesList; 

      /** @brief Problem setup **/ 
      void problemSetup( const ProblemSpecP& db ); 
      
      /** @brief Compute the properties/timestep **/ 
      void sched_compute_radiation_properties( const LevelP& level, SchedulerP& sched, const MaterialSet* matls, 
                                               const int time_substep, const bool doing_initialization ); 

      /** @brief see sched_compute_radiation_properties **/ 
      void compute_radiation_properties( const ProcessorGroup* pc, 
                                         const PatchSubset* patches, 
                                         const MaterialSubset* matls, 
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw, 
                                         const int time_substep, 
                                         const bool doing_initialization);

      class PropertyCalculatorBase { 

        public: 
          PropertyCalculatorBase() {}
          virtual ~PropertyCalculatorBase(){

            VarLabel::destroy(_abskg_label); 
          }

          virtual bool problemSetup( const ProblemSpecP& db )=0; 
          virtual void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, 
                                     RadCalcSpeciesList species, constCCVariable<double>& mixT,  
                                     CCVariable<double>& abskg )=0; 


          virtual std::vector<std::string> get_sp() = 0;

          inline const VarLabel* get_abskg_label() { return _abskg_label; } 


          std::string get_abskg_name(){ return _abskg_name;}

          /** @brief This function sums in the particle contribution to the gas contribution **/ 
          template <class T> 
          void sum_abs( CCVariable<double>& absk_tot, T& abskp, const Patch* patch ){ 
            for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

              absk_tot[*iter] += abskp[*iter];  
              
            }
          }

        protected: 

          const VarLabel* _abskg_label;   // gas absorption coefficient

          std::string _abskg_name; 



      };

      typedef std::vector<PropertyCalculatorBase*> CalculatorVec;
      CalculatorVec _all_calculators; 

      //______________________________________________________________________
      //
      class ConstantProperties : public PropertyCalculatorBase  { 

        public: 
          ConstantProperties();
          ~ConstantProperties();
          
          bool problemSetup( const ProblemSpecP& db ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }

        private: 
          double _abskg_value; 


      }; 
      //______________________________________________________________________
      //
      class specialProperties : public PropertyCalculatorBase  { 

        public: 
          specialProperties();
          ~specialProperties();
          
          bool problemSetup( const ProblemSpecP& db ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }

        private: 
          double _expressionNumber; 


      }; 
      //______________________________________________________________________
      //
      class  BurnsChriston : public PropertyCalculatorBase  { 

        public: 
          BurnsChriston();
          ~BurnsChriston();
          bool problemSetup( const ProblemSpecP& db ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp(){
            std::vector<std::string> void_vec; 
            return void_vec; 
          }
        private: 
          double _value;
          Point _notSetMin;
          Point _notSetMax;
          Point _min;
          Point _max;
      }; 
      //______________________________________________________________________
      //
      class HottelSarofim : public PropertyCalculatorBase  { 

        public: 
          HottelSarofim();
          ~HottelSarofim();
          
          bool problemSetup( const ProblemSpecP& db ); 
          void compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg );

          std::vector<std::string> get_sp();

        private: 

          std::string _co2_name;                     ///< table name
          std::string _h2o_name;                     ///< table name 
          std::string _soot_name;                    ///< property name
          double d_opl;                              ///< optical length; 
      }; 

#ifdef HAVE_RADPROPS
      //______________________________________________________________________
      //
      class RadPropsInterface : public PropertyCalculatorBase  { 

        public: 
          RadPropsInterface();
          ~RadPropsInterface(); 
          bool problemSetup( const ProblemSpecP& db );
          void compute_abskg( const Patch* patch, 
              constCCVariable<double>& VolFractionBC, 
              RadCalcSpeciesList species,  
              constCCVariable<double>& mixT, 
              CCVariable<double>& abskg); 

          std::vector<std::string> get_sp(){ return _species; }

        private: 

          GreyGas* _gg_radprops; 
          std::vector<std::string> _species;               // to match the Arches varlabels
          std::vector<RadiativeSpecies> _radprops_species; // for rad props
          std::string _mix_mol_weight_name; 
          std::vector<double> _sp_mw; 

      }; 
#endif

  // - The base class opticalPropertyCalculatorBase is intended to create a set
  // - of modular functions for computing optical properties for various particle
  // - types.  i.e. coal, glass, metal-oxides, etc.

      class opticalPropertyCalculatorBase { 

        public: 
          opticalPropertyCalculatorBase() {}
          virtual ~opticalPropertyCalculatorBase(){

            VarLabel::destroy(_abskp_label);  
            for (int i=0; i< _nQn_part; i++){
              VarLabel::destroy(_abskp_label_vector[i]);
            }

          }

          virtual bool problemSetup( Task* tsk, int time_substep)=0; 

          virtual void computeComplexIndex( const Patch* patch,
                                            constCCVariable<double>& VolFractionBC,
                                            SCIRun::StaticArray < constCCVariable<double> > &composition,
                                            SCIRun::StaticArray < CCVariable<double> > &complexReal )=0;

          virtual void computeAsymmetryFactor( const Patch* patch,
                                               constCCVariable<double>& VolFractionBC,
                                               SCIRun::StaticArray < CCVariable<double> > &scatktQuad,
                                               SCIRun::StaticArray < constCCVariable<double> > &composition, 
                                               CCVariable<double>& scatkt,  
                                               CCVariable<double>  &asymmetryParam )=0;

          virtual void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                      RadCalcSpeciesList size, RadCalcSpeciesList pT, 
                                      RadCalcSpeciesList weights, 
                                      const int Nqn, CCVariable<double>& abskpt, 
                                      SCIRun::StaticArray < CCVariable<double> >  &abskp,
                                      SCIRun::StaticArray < CCVariable<double> >  &complexReal)=0;

          virtual void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                       RadCalcSpeciesList size, RadCalcSpeciesList pT, 
                                       RadCalcSpeciesList weights, 
                                       const int Nqn, CCVariable<double>& scatkt, 
                                       SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                       SCIRun::StaticArray < CCVariable<double> > &complexReal)=0;


          //virtual std::vector<std::string> get_sp() = 0;

          inline const VarLabel* get_abskp_label() { return _abskp_label; }
          inline const VarLabel* get_scatkt_label(){ return _scatkt_label;}
          inline std::vector<const VarLabel*> get_complexIndexReal_label() { return _complexIndexReal_label; } 
          inline std::vector<const VarLabel*> get_abskp_label_vector() { return _abskp_label_vector; } 
          inline const VarLabel* get_asymmetryParam_label() { return _asymmetryParam_label; }

          //[>* @brief Matches label names to labels *<] 
          std::string get_complexIndexReal_name(){ return _complexIndexReal_name;}
          std::string get_asymmetryParam_name(){ return _asymmetryParam_name;}
          std::string get_abskp_name(){ return _abskp_name;}
          std::string get_scatkt_name(){return _scatkt_name;}
          bool get_complexIndexBool(){return _computeComplexIndex;}

          inline std::vector< const VarLabel*> getRequiresLabels(){
            return _compositionLabels;
          }


      bool construction_success;
        protected:
      int _ncomp ;                                // number of components
      std::vector< const VarLabel*> _compositionLabels;
      std::string _asymmetryParam_name;
      std::string _complexIndexReal_name;
      const VarLabel* _asymmetryParam_label;   // gas absorption coefficient
      const VarLabel* _abskp_label;   // gas absorption coefficient
      const VarLabel* _scatkt_label;   // gas absorption coefficient
      std::vector< const VarLabel*> _abskp_label_vector;
      std::vector< const VarLabel*> _complexIndexReal_label;   // particle absorption coefficient
      std::complex<double> _HighComplex;
      std::complex<double> _LowComplex;
      std::string _abskp_name; 
      std::string _scatkt_name; 
      bool _scatteringOn;  // local not needed for scattering, because it is always local as of 11-2014
      int _nQn_part ;                                // number of quadrature nodes in DQMOM
      int _computeComplexIndex ; // 


      };

      class coalOptics : public opticalPropertyCalculatorBase  { 

        public: 
          coalOptics(const ProblemSpecP& db, bool scatteringOn);
          ~coalOptics(); 

          bool problemSetup(Task* tsk, int time_substep );

          void computeComplexIndex( const Patch* patch,
                                    constCCVariable<double>& VolFractionBC,
                                    SCIRun::StaticArray<constCCVariable<double> > &composition, 
                                    SCIRun::StaticArray < CCVariable<double> > &complexReal);


          void computeAsymmetryFactor( const Patch* patch,
                                       constCCVariable<double>& VolFractionBC,
                                       SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                       SCIRun::StaticArray < constCCVariable<double> > &composition,
                                       CCVariable<double>& scatkt,
                                       CCVariable<double>  &asymmetryParam);

          virtual void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                      RadCalcSpeciesList size, RadCalcSpeciesList pT, 
                                      RadCalcSpeciesList weights, 
                                      const int Nqn, CCVariable<double>& abskpt, 
                                      SCIRun::StaticArray < CCVariable<double> >  &abskp,
                                      SCIRun::StaticArray < CCVariable<double> >  &complexReal);

          virtual void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                       RadCalcSpeciesList size, RadCalcSpeciesList pT, 
                                       RadCalcSpeciesList weights, 
                                       const int Nqn, CCVariable<double>& scatkt, 
                                       SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                       SCIRun::StaticArray < CCVariable<double> > &complexReal);



        private: 

          double _rawCoalReal;
          double _rawCoalImag;
          double _charReal;
          double _charImag;
          double _ashReal;
          double _ashImag;
          std::complex<double> _complexLo;  
          std::complex<double> _complexHi;  
          ParticleRadCoeffs3D* _part_radprops; 

          

          double  _charAsymm;
          double  _rawCoalAsymm;
          double  _ashAsymm;

          std::vector<double>  _ash_mass;        /// particle sizes in diameters
          std::vector< std::string >  _composition_names ;

          bool _p_planck_abskp; 
          bool _p_ros_abskp; 
      }; 


      class basic : public opticalPropertyCalculatorBase  { 

        public: 
          basic(const ProblemSpecP& db, bool scatteringOn);
          ~basic(); 

          bool problemSetup(Task* tsk, int time_substep );

          void computeComplexIndex( const Patch* patch,
                                    constCCVariable<double>& VolFractionBC,
                                    SCIRun::StaticArray<constCCVariable<double> > &composition, 
                                    SCIRun::StaticArray < CCVariable<double> > &complexReal);


          void computeAsymmetryFactor( const Patch* patch,
                                       constCCVariable<double>& VolFractionBC,
                                       SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                       SCIRun::StaticArray < constCCVariable<double> > &composition,
                                       CCVariable<double>& scatkt,
                                       CCVariable<double>  &asymmetryParam);

          virtual void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                      RadCalcSpeciesList size, RadCalcSpeciesList pT,  
                                      RadCalcSpeciesList weights, 
                                      const int Nqn, CCVariable<double>& abskpt, 
                                      SCIRun::StaticArray < CCVariable<double> >  &abskp,
                                      SCIRun::StaticArray < CCVariable<double> >  &complexReal);

          virtual void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                       RadCalcSpeciesList size, RadCalcSpeciesList pT, 
                                       RadCalcSpeciesList weights, 
                                       const int Nqn, CCVariable<double>& scatkt, 
                                       SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                       SCIRun::StaticArray < CCVariable<double> > &complexReal);



        private: 

          double _Qabs ;  // This is a fudge factor for particle absorption coefficients, used by Julien in a coal model
      }; 

      class constantCIF : public opticalPropertyCalculatorBase  { 

        public: 
          constantCIF(const ProblemSpecP& db, bool scatteringOn);
          ~constantCIF(); 

          bool problemSetup(Task* tsk, int time_substep );

          void computeComplexIndex( const Patch* patch,
                                    constCCVariable<double>& VolFractionBC,
                                    SCIRun::StaticArray<constCCVariable<double> > &composition, 
                                    SCIRun::StaticArray < CCVariable<double> > &complexReal);


          void computeAsymmetryFactor( const Patch* patch,
                                       constCCVariable<double>& VolFractionBC,
                                       SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                       SCIRun::StaticArray < constCCVariable<double> > &composition,
                                       CCVariable<double>& scatkt,
                                       CCVariable<double>  &asymmetryParam);

          virtual void compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                      RadCalcSpeciesList size, RadCalcSpeciesList pT, 
                                      RadCalcSpeciesList weights, 
                                      const int Nqn, CCVariable<double>& abskpt, 
                                      SCIRun::StaticArray < CCVariable<double> >  &abskp,
                                      SCIRun::StaticArray < CCVariable<double> >  &complexReal);

          virtual void compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                       RadCalcSpeciesList size, RadCalcSpeciesList pT, 
                                       RadCalcSpeciesList weights, 
                                       const int Nqn, CCVariable<double>& scatkt, 
                                       SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                       SCIRun::StaticArray < CCVariable<double> > &complexReal);



        private: 

          ParticleRadCoeffs* _part_radprops; 
          double _constAsymmFact;
          bool _p_planck_abskp; 
          bool _p_ros_abskp; 
      }; 





    private: 

      const int _matl_index; 
      std::string _temperature_name; 
      const VarLabel* _temperature_label; 


  }; 
} 

#endif



