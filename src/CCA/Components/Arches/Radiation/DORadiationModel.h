/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Util/Timers/Timers.hpp>


namespace Uintah {

  class ArchesLabel;

class DORadiationModel{

public:

      RadiationSolver* d_linearSolver;

      DORadiationModel(const ArchesLabel* label,
                       const MPMArchesLabel* MAlab,
                       const ProcessorGroup* myworld,
                       bool sweepMethod);


      virtual ~DORadiationModel();


      virtual void problemSetup(ProblemSpecP& params);

      virtual void boundarycondition(const ProcessorGroup* pc,
                                     const Patch* patch,
                                     CellInformation* cellinfo, 
                                     ArchesVariables* vars,
                                     ArchesConstVariables* constvars);

      virtual void intensitysolve(const ProcessorGroup* pc,
                                  const Patch* patch,
                                  CellInformation* cellinfo, 
                                  ArchesVariables* vars,
                                  ArchesConstVariables* constvars, 
                                  CCVariable<double>& divQ,
                                  int wall_type, int matlIndex, DataWarehouse* new_dw, DataWarehouse* old_dw,
                                  bool old_DW_isMissingIntensities);

      void intensitysolveSweepOptimized(const Patch* patch,
                               int matlIndex,
                               DataWarehouse* new_dw, 
                               DataWarehouse* old_dw,
                               int cdirecn);

      void intensitysolveSweepOptimizedOLD(const Patch* patch,
                               int matlIndex,
                               DataWarehouse* new_dw, 
                               DataWarehouse* old_dw,
                               int cdirecn);


      void setExtraSweepingLabels(int nphase);

      void getDOSource(const Patch* patch,
                       int matlIndex,                    
                       DataWarehouse* new_dw,            
                       DataWarehouse* old_dw);           


      void computeFluxDiv(const Patch* patch,
                          int matlIndex,  
                          DataWarehouse* new_dw, 
                          DataWarehouse* old_dw);

      void setIntensityBC(const Patch* patch,
                                  int matlIndex,  
                                  CCVariable<double>& intensity,
                                  constCCVariable<double>& radTemp,
                                  constCCVariable<int>& cellType,
                                  int iSpectralBand=0);

      void  setIntensityBC2Orig(const Patch* patch,
                                int matlIndex,  
                                DataWarehouse* new_dw, 
                                DataWarehouse* old_dw, int ix);

      int getIntOrdinates();

      bool reflectionsBool();

      bool needIntensitiesBool();

      bool ScatteringOnBool();

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
        return _nQn_part;
      }
      inline int xDir( int ix){
        return (int) _plusX[ix];
      }
      inline int yDir( int ix){
        return (int) _plusY[ix];
      }
      inline int zDir( int ix){
        return (int) _plusZ[ix];
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
        return d_nbands;
      }

private:

      std::vector<double> _grey_reference_weight;
      double _nphase; // optical length
      double _solve_start;
      double d_opl; // optical length
      const ArchesLabel*    d_lab;
      const MPMArchesLabel* d_MAlab;
      const ProcessorGroup* d_myworld;
      const PatchSet* d_perproc_patches;
      
      int d_sn, d_totalOrds; // totalOrdinates = sn*(sn+2)
      std::string d_quadratureSet;                // Name of Method used to determine intensity directions

      void computeOrdinatesOPL();
      int d_lambda;
      const int ffield;

      std::vector< std::vector < std::vector < Ghost::GhostType > > > _gv;
   
      OffsetArray1<double> oxi;
      OffsetArray1<double> omu;
      OffsetArray1<double> oeta;
      OffsetArray1<double> wt;

      std::vector < bool >  _plusX;
      std::vector < bool >  _plusY;
      std::vector < bool >  _plusZ;
      std::vector < int >  xiter;
      std::vector < int >  yiter;
      std::vector < int >  ziter;

      OffsetArray1<double> rgamma;
      OffsetArray1<double> sd15;
      OffsetArray1<double> sd;
      OffsetArray1<double> sd7;
      OffsetArray1<double> sd3;

      OffsetArray1<double> srcbm;
      OffsetArray1<double> srcpone;
      OffsetArray1<double> qfluxbbm;

      double d_xfluxAdjust;
      double d_yfluxAdjust;
      double d_zfluxAdjust;

      bool d_print_all_info; 
      bool reflectionsTurnedOn;
      bool _scatteringOn;
      bool _usePreviousIntensity;
      bool _zeroInitialGuess;
      bool _radiateAtGasTemp; // this flag is arbitrary for no particles
      int _sweepMethod;
      int d_nbands{1};
      bool _LspectralSolve;

      const VarLabel* _scatktLabel;
      const VarLabel* _asymmetryLabel;
      const VarLabel*  _abskt_label;
      const VarLabel*  _T_label;
      const VarLabel*  _cellTypeLabel;
      const VarLabel* _fluxE;
      const VarLabel* _fluxW;
      const VarLabel* _fluxN;
      const VarLabel* _fluxS;
      const VarLabel* _fluxT;
      const VarLabel* _fluxB;
      const VarLabel* _volQ;
      const VarLabel* _divQ;
      Timers::Simple _timer;

      std::vector< const VarLabel*> _IntensityLabels;
      std::vector< const VarLabel*> _emiss_plus_scat_source_label; // for sweeps, needed because Intensities fields are solved in parallel

      std::vector< std::vector< const VarLabel*> > _patchIntensityLabels; 
      std::vector< const VarLabel*> _radiationFluxLabels;

      std::vector< std::vector < double > > cosineTheta;
      std::vector < double >  solidAngleWeight;

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

      std::vector<const VarLabel*>  _radIntSource;
      std::vector<std::string> _radIntSource_names;


      std::vector<std::string> _abskg_name_vector;
      std::vector<std::string> _abswg_name_vector;

      std::vector< const VarLabel*> _abskg_label_vector;
      std::vector< const VarLabel*> _abswg_label_vector;

      // variables needed for particles
      std::vector<std::string> _temperature_name_vector;
      std::vector<std::string> _abskp_name_vector;

      std::vector< const VarLabel*> _abskp_label_vector;
      std::vector< const VarLabel*> _temperature_label_vector;
      int _nQn_part ;                                // number of quadrature nodes in DQMOM
      double _sigma;


}; // end class RadiationModel

} // end namespace Uintah

#endif
