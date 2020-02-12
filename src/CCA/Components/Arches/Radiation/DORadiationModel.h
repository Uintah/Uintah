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

//----- DORadiationModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_DORadiationModel_h
#define Uintah_Component_Arches_DORadiationModel_h

/***************************************************************************
CLASS
    DORadiationModel
       Sets up the DORadiationModel

GENERAL INFORMATION
    DORadiationModel.h - Declaration of DORadiationModel class

    Author:Gautham Krishnamoorthy (gautham@crsim.utah.edu)
           Rajesh Rawat (rawat@crsim.utah.edu)

    Creation Date : 06-18-2002

    C-SAFE


***************************************************************************/
#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Util/Timers/Timers.hpp>


namespace Uintah {

  class ApplicationInterface;

  class ArchesLabel;
  class MPMArchesLabel;

class DORadiationModel{

public:

      RadiationSolver* d_linearSolver;

      DORadiationModel(const ArchesLabel* label,
                       const MPMArchesLabel* MAlab,
                       const ProcessorGroup* myworld,
                       bool sweepMethod);


      virtual ~DORadiationModel();


      virtual void problemSetup(ProblemSpecP& params);

      // A pointer to the application so to get a handle to the
      // performanance stats.
      virtual void setApplicationInterface(ApplicationInterface * app) {
        m_application = app;
      };

      virtual void boundarycondition(const ProcessorGroup* pc,
                                     const Patch* patch,
                                     CellInformation* cellinfo,
                                     ArchesVariables* vars,
                                     ArchesConstVariables* constvars){};

      virtual void intensitysolve(const ProcessorGroup* pc,
                                  const Patch* patch,
                                  CellInformation* cellinfo,
                                  ArchesVariables* vars,
                                  ArchesConstVariables* constvars,
                                  CCVariable<double>& divQ,
                                  int wall_type,
                                  int matlIndex,
                                  DataWarehouse* new_dw,
                                  DataWarehouse* old_dw,
                                  bool old_DW_isMissingIntensities);

      void intensitysolveSweepOptimized(const Patch* patch,
                                        int matlIndex,
                                        DataWarehouse* new_dw,
                                        DataWarehouse* old_dw,
                                        int cdirecn);

      void getDOSource(const Patch* patch,
                       const int matlIndex,
                       DataWarehouse* new_dw,
                       DataWarehouse* old_dw);


      void computeFluxDiv(const Patch* patch,
                          const int matlIndex,
                          DataWarehouse* new_dw,
                          DataWarehouse* old_dw);

      void  setIntensityBC(const Patch* patch,
                           const int matlIndex,
                           DataWarehouse* new_dw,
                           DataWarehouse* old_dw,
                           const Ghost::GhostType me,
                           const int ord);

      // returns the total number of directions, sn*(sn+2)
      int getIntOrdinates(){
        return m_totalOrds;
      }

      bool reflectionsBool(){
        return m_doReflections;
      }

      bool needIntensitiesBool();

      // Model scattering physics of particles?
      bool ScatteringOnBool(){
        return m_doScattering;
      };

      void setLabels( const VarLabel* abskg ,
                      const VarLabel* abskt,
                      const VarLabel* T_label,
                      const VarLabel* cellType,

    std::vector<const VarLabel* > radIntSource,
                      const VarLabel*  FluxE,
                      const VarLabel*  FluxW,
                      const VarLabel*  FluxN,
                      const VarLabel*  FluxS,
                      const VarLabel*  FluxT,
                      const VarLabel*  FluxB,
                      const VarLabel*  volQ,
                      const VarLabel*  divQ);


      void setLabels(   );

      inline std::vector< const VarLabel*> getAbskpLabels(){
        return _abskp_label_vector;
      }

      inline std::vector< const VarLabel*> getPartTempLabels(){
        return _temperature_label_vector;
      }

      inline int get_nQn_part(){
        return m_nQn_part;
      }
      inline int xDir( int ix){
        return   m_plusX[ix];
      }
      inline int yDir( int ix){
        return  m_plusY[ix] ;
      }
      inline int zDir( int ix){
        return  m_plusZ[ix] ;
      }

      std::vector<std::string> gasAbsorptionNames(){
        return _abskg_name_vector;
      }
      std::vector<std::string> gasWeightsNames(){
        return _abswg_name_vector;
      }
      inline std::vector< const VarLabel*> getAbskgLabels(){
        return _abskg_label_vector;
      }

      inline std::vector< const VarLabel*> getAbswgLabels(){
        return _abswg_label_vector;
      }

      inline int spectralBands(){
        return m_nbands;
      }

      inline bool spectralSootOn(){
        return _LspectralSootOn;
      }

private:

      //__________________________________
      //  Variables
      enum initGuess{ NONE, ZERO, OLD_INTENSITY};

      const ProcessorGroup* d_myworld;
      const Ghost::GhostType m_gn = Ghost::None;
      
      std::vector<double> _grey_reference_weight;
      double _nphase;                               // optical length
      std::vector< std::vector < std::vector < Ghost::GhostType > > > _gv;

      std::string m_quadratureSet;                 // Name of Method used to determine intensity directions
      
      const int m_lambda{1};                       //WARNING: HARDCODED.
      const int m_ffield;


      std::vector<double> m_omu;
      std::vector<double> m_oeta;
      std::vector<double> m_oxi;
      std::vector<double> m_wt;
  
      std::vector<bool>  m_plusX;     // What are these?
      std::vector<bool>  m_plusY;
      std::vector<bool>  m_plusZ;
      
      std::vector<int>  m_xiter;
      std::vector<int>  m_yiter;
      std::vector<int>  m_ziter;

      double m_xfluxAdjust;
      double m_yfluxAdjust;
      double m_zfluxAdjust;
      
      // switches and flags
      bool _LspectralSolve      {false};
      bool _LspectralSootOn     {false};
      bool m_print_all_info     {false};
      bool m_radiateAtGasTemp   {true};                 // this flag is arbitrary for no particles
      bool m_doReflections      {false};
      bool m_doScattering       {false};
      bool m_addOrthogonalDirs {false};
      initGuess m_initialGuess  {NONE};
      int _sweepMethod;
      
      // looping limits and physical constants
      int m_nbands{1};
      int m_nQn_part;      
      int m_sn;
      int m_totalOrds;                                  // totalOrdinates = sn*(sn+2)
      const double _sigma{5.67e-8};                     //  w / m^2 k^4


      std::vector< std::vector<double>> m_cosineTheta;
      std::vector<double>  m_solidAngleWeight;
      Timers::Simple _timer;
      
      // A pointer to the application so to get a handle to the performanance stats.
      ApplicationInterface* m_application{nullptr};
      
      //__________________________________
      //  VarLabels
      const VarLabel* _abskt_label;
      const VarLabel* _asymmetry_label;
      const VarLabel* _cellType_label;
      const VarLabel* _divQ_label;
      const VarLabel* _fluxB_label;
      const VarLabel* _fluxE_label;
      const VarLabel* _fluxN_label;
      const VarLabel* _fluxS_label;
      const VarLabel* _fluxT_label;
      const VarLabel* _fluxW_label;
      const VarLabel* _scatkt_label;
      const VarLabel* _T_label;
      const VarLabel* _volQ_label;

      std::vector< const VarLabel*> _abskg_label_vector;
      std::vector< const VarLabel*> _abskp_label_vector;
      std::vector< const VarLabel*> _abswg_label_vector;
      std::vector< const VarLabel*> _emiss_plus_scat_source_label; // for sweeps, needed because Intensities fields are solved in parallel
      std::vector< const VarLabel*> _IntensityLabels;
      std::vector< const VarLabel*> _radiationFluxLabels;
      std::vector< const VarLabel*> _emissSrc_label;
      std::vector< const VarLabel*> _temperature_label_vector;
      std::vector< std::vector< const VarLabel*> > _patchIntensityLabels;

      //__________________________________
      // VarLabel names
      std::vector<std::string> _abskg_name_vector;
      std::vector<std::string> _abswg_name_vector;

      // variables needed for particles
      std::vector<std::string> _temperature_name_vector;
      std::vector<std::string> _abskp_name_vector;

      //__________________________________
      //  methods
     void insertEveryNth( const std::vector<std::vector<double>>& orthogonalCosineDirs,
                          const int nthElement,
                          const int dir,
                          std::vector<double>& vec);

      void computeOrdinatesOPL();
      
      template<class TYPE>
      void computeScatteringIntensities(int direction,
                                        constCCVariable<double> &scatkt,
                                        std::vector< TYPE > &Intensities,
                                        CCVariable<double> &scatIntensitySource,
                                        constCCVariable<double> &asymmetryFactor,
                                        const Patch* patch);


      void computeIntensitySource( const Patch* patch,
                                   std::vector <constCCVariable<double> >&abskp,
                                   std::vector <constCCVariable<double> > &pTemp,
                                   std::vector <constCCVariable<double> > &abskg,
                                   constCCVariable<double>  &gTemp,
                                   std::vector <CCVariable<double> >&b_sourceArray,
                                   std::vector <constCCVariable<double> >&spectral_weights);
                                   
    int intensityIndx(const int ord,
                      const int iband);

      

}; // end class RadiationModel

} // end namespace Uintah

#endif
