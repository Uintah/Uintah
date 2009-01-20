//----- BackwardMCRTSolver.h  --------------------------------------------------

#ifndef Uintah_Component_Arches_BackwardMCRTSolver_h
#define Uintah_Component_Arches_BackwardMCRTSolver_h

/**
* @class BackwardMCRTSolve
* @author Xiaojing Sun
* @date Dec 11, 2008
*
* @brief Backward(Reverse) MonteCarlo Ray-Tracing Model for Radiation Heat Transfer
*
*
*/


#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/PropertyModel.h>

namespace Uintah {
  
  class BoundaryCondition;
  class PropertyModel;
  
  class BackwardMCRTSolver: public RadiationModel {
    
  public:
    
    
    inline BackwardMCRTSolver(BoundaryCondition* bndry_cond, const ProcessorGroup* myworld);
    
    inline virtual ~BackwardMCRTSolver();
    
    /** @brief Set any parameters from input file, initialize any constants, etc.. */
    virtual void problemSetup(const ProblemSpecP& params);
    
    
    virtual void boundarycondition(const ProcessorGroup* pc,
				   const Patch* patch,
				   CellInformation* cellinfo, 
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars);
    
    // RadiationModel.cc 's function is intensitysolve
    virtual void intensitySolve(const ProcessorGroup* pc,
				const Patch* patch,
				CellInformation* cellinfo, 
				ArchesVariables* vars,
				ArchesConstVariables* constvars);
    
    
    // Schedule it when you need to read from DatawareHouse?
    void sched_initRealSurfOutInten();
    
    /** @brief initialize real surface (boundary)'s outgoing intensity and blackbody intensity */
    void initRealSurfOutInten();
    
    void sched_initVolOutInten();
    
    /** @brief initialize volume (media)'s outgoing blackbody intensity and intensity */
    void initVolOutInten();
    
    void computeIncomingInten();
    
    void computeNetInten();
    
    void computeHeatFlux();
    
    void computeDivq();
    
  protected: 
    // boundary condition
    BoundaryCondition* d_boundaryCondition;
    
  private:
    
    double d_xumax;
    
    void computeOrdinatesOPL();
    
    int MAXITR;
    double QACCU, d_opl, af, qerr, totsrc;
    int iflag, iriter;
    int lambda;
    double wavemin, wavemax, dom, omega, srcsum;
    int ffield;
    int wall;
    int symtry;
    int pfield;
    int sfield;
    int pbcfld;
    int outletfield;
    bool d_SHRadiationCalc, lprobone, lprobtwo, lprobthree, lradcal, lwsgg, lplanckmean, lpatchmean;
    
    OffsetArray1<double> fraction;
      OffsetArray1<double> fractiontwo;
    
    //      OffsetArray1<double> ord;
    OffsetArray1<double> oxi;
    OffsetArray1<double> omu;
    OffsetArray1<double> oeta;
    OffsetArray1<double> wt;
    OffsetArray1<double> arean;
    OffsetArray1<double> areatb;
    
    OffsetArray1<double> rgamma;
    OffsetArray1<double> sd15;
    OffsetArray1<double> sd;
    OffsetArray1<double> sd7;
    OffsetArray1<double> sd3;
    
    OffsetArray1<double> srcbm;
    OffsetArray1<double> srcpone;
    OffsetArray1<double> qfluxbbm;
    const ProcessorGroup* d_myworld;
    
    
  }; // end class RadiationModel
  
} // end namespace Uintah

#endif




