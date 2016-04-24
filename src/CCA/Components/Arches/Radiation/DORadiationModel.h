/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <Core/Containers/StaticArray.h>

namespace Uintah {

  class ArchesLabel;

class DORadiationModel{

public:

      RadiationSolver* d_linearSolver;

      DORadiationModel(const ArchesLabel* label,
                       const MPMArchesLabel* MAlab,
                       const ProcessorGroup* myworld);


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
      int getIntOrdinates();

      bool reflectionsBool();

      bool needIntensitiesBool();

      bool ScatteringOnBool();

      void setLabels() ;

      inline std::vector< const VarLabel*> getAbskpLabels(){
        return _abskp_label_vector;
      }

      inline std::vector< const VarLabel*> getPartTempLabels(){
        return _temperature_label_vector;
      }

      inline int get_nQn_part(){
        return _nQn_part;
      }

private:

      double d_opl; // optical length
      const ArchesLabel*    d_lab;
      const MPMArchesLabel* d_MAlab;
      const ProcessorGroup* d_myworld;
      const PatchSet* d_perproc_patches;
      
      int d_sn, d_totalOrds; // totalOrdinates = sn*(sn+2)

      void computeOrdinatesOPL();
      int d_lambda;
      int ffield;

      OffsetArray1<double> fraction;

      OffsetArray1<double> oxi;
      OffsetArray1<double> omu;
      OffsetArray1<double> oeta;
      OffsetArray1<double> wt;

      OffsetArray1<double> rgamma;
      OffsetArray1<double> sd15;
      OffsetArray1<double> sd;
      OffsetArray1<double> sd7;
      OffsetArray1<double> sd3;

      OffsetArray1<double> srcbm;
      OffsetArray1<double> srcpone;
      OffsetArray1<double> qfluxbbm;

      bool d_print_all_info; 
      bool reflectionsTurnedOn;
      bool _scatteringOn;
      bool _usePreviousIntensity;
      bool _zeroInitialGuess;
      bool _radiateAtGasTemp; // this flag is arbitrary for no particles

      const VarLabel* _scatktLabel;
      const VarLabel* _asymmetryLabel;

      std::vector< const VarLabel*> _IntensityLabels;
      std::vector< const VarLabel*> _radiationFluxLabels;

      std::vector< std::vector < double > > cosineTheta;
      std::vector< std::vector < double > > solidAngleQuad;

      template<class TYPE> 
      void computeScatteringIntensities(int direction,
                                        constCCVariable<double> &scatkt,
                                        StaticArray< TYPE > &Intensities,
                                        CCVariable<double> &scatIntensitySource,
                                        constCCVariable<double> &asymmetryFactor,
                                        const Patch* patch);



      void computeIntensitySource( const Patch* patch,
				   StaticArray <constCCVariable<double> >&abskp,
				   StaticArray <constCCVariable<double> > &pTemp,
				   constCCVariable<double>  &abskg,
				   constCCVariable<double>  &gTemp,
				   CCVariable<double> &b_sourceArray);

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
