
#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H

#include <Uintah/Interface/CFDInterface.h>
class ProcessorContext;
class Region;

namespace Uintah {
namespace Components {

using Uintah::Interface::CFDInterface;
using Uintah::Interface::DataWarehouseP;
using Uintah::Interface::SchedulerP;
using Uintah::Interface::ProblemSpecP;
using Uintah::Grid::LevelP;
using Uintah::Grid::GridP;
using Uintah::Grid::VarLabel;

class ICE : public CFDInterface {
public:
    ICE();
    virtual ~ICE();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid);
   virtual void scheduleInitialize(const LevelP& level,
				   SchedulerP&,
				   DataWarehouseP&);
	 
    virtual void scheduleComputeStableTimestep(const LevelP&,
					       SchedulerP&,
					       const VarLabel*,
					       DataWarehouseP&);
    void actuallyComputeStableTimestep(const ProcessorContext*,
				       const Region* region,
				       const DataWarehouseP&,
				       DataWarehouseP&);
    virtual void scheduleTimeAdvance(double t, double dt, const LevelP&, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&);
    void actuallyTimeStep(const ProcessorContext*,
			  const Region* region,
			  const DataWarehouseP&,
			  DataWarehouseP&);

private:
    // These two will go away SOON - a really bad habit, won't work in parallel, blah blah blah
    double cheat_t, cheat_delt;
    double
            *R,                         /* gas constant R[material]          */ 
             ***x_CC,                   /* x-coordinate of cell center      */
             ***y_CC,                   /* y-coordinate of cell center      */
             ***z_CC,                   /* z-coordinate of cell center      */
             ***Vol_CC,                 /* vol of the cell at the cellcenter*/
                                        /* (x, y, z)                        */
            /*------to be treated as pointers---*/
             *****x_FC,                 /* x-coordinate of face center      */
                                        /* x_FC(i,j,k,face)                 */
                                        /* cell i,j,k                       */
             *****y_FC,                 /* y-coordinate of face center      */
                                        /* y_FC(i,j,k,face)                 */
                                        /* of cell i,j,k                    */
             *****z_FC;                 /* z-coordinate of face center      */
                                        /* z_FC(i,j,k,face)                 */
                                        /* of cell i,j,k                    */
            /*----------------------------------*/ 

/* ______________________________   
*  Cell-centered and Face centered                      
* ______________________________   */ 
    double                              /* (x,y,z,material                  */
            ****uvel_CC,                /* u-cell-centered velocity         */
            ****vvel_CC,                /*  v-cell-centered velocity        */
            ****wvel_CC,                /* w cell-centered velocity         */
            ****delPress_CC,            /* cell-centered change in pressure */                                                                                       
            ****press_CC,               /* Cell-centered pressure           */
            ****Temp_CC,                /* Cell-centered Temperature        */
            ****rho_CC,                 /* Cell-centered density            */
            ****viscosity_CC,           /* Cell-centered Viscosity          */ 
            ****thermalCond_CC,         /* Cell-centered thermal conductivity*/
            ****cv_CC,                  /* Cell-centered specific heat      */ 
            ****mass_CC,                /* total mass, cell-centered        */
            ****xmom_CC,                /* x-dir momentum cell-centered     */
            ****ymom_CC,                /* y-dir momentum cell-centered     */
            ****zmom_CC,                /* z-dir momentum cell-centered     */
            ****int_eng_CC,             /* Internal energy cell-centered    */
            ****total_eng_CC,           /* Total energy cell-centered       */  
            ****scalar1_CC,             /* Cell-centered scalars            */   
            ****scalar2_CC,             /* (x, y, z, material)              */
            ****scalar3_CC,
            /*------to be treated as pointers---*/
                                        /*______(x,y,z,face, material)______*/
            ******uvel_FC,              /* u-face-centered velocity         */
            ******vvel_FC,              /* *v-face-centered velocity        */
            ******wvel_FC,              /* w face-centered velocity         */
            ******press_FC,             /* face-centered pressure           */
            ******tau_x_FC,             /* *x-stress component at each face */
            ******tau_y_FC,             /* *y-stress component at each face */
            ******tau_z_FC,             /* *z-stress component at each face */            
            /*----------------------------------*/                                              
           *grav,                       /* gravity (dir)                    */
                                        /* x-dir = 1, y-dir = 2, z-dir = 3  */
            ****speedSound;             /* speed of sound (x,y,z, material) */    
             
/* ______________________________   
*  Lagrangian Variables            
* ______________________________   */ 
    double                              /*_________(x,y,z,material)_________*/
            ****rho_L_CC,               /* Lagrangian cell-centered density */
            ****mass_L_CC,              /* Lagrangian cell-centered mass    */
            ****Temp_L_CC,              /* Lagrangian cell-centered Temperature */
            ****press_L_CC,             /* Lagrangian cell-centered pressure*/            
            ****xmom_L_CC,              /* Lagrangian cell-centered momentum*/
            ****ymom_L_CC,              /* Lagrangian cell-centered momentum*/
            ****zmom_L_CC,              /* Lagrangian cell-centered momentum*/
            ****int_eng_L_CC,           /* Lagrangian cc internal energy    */
            ****Vol_L_CC,               /* Lagrangian cell-centered volume  */
/*__________________________________
* source terms
*___________________________________*/
            ****mass_source,            /* Mass source term (x,y,z, material */
            ****xmom_source,            /* momentum source terms            */
                                        /* (x, y, z, material)              */
            ****ymom_source,
            ****zmom_source,
            ****int_eng_source,         /* internal energy source           */
/*__________________________________
*   MISC Variables
*___________________________________*/            
            ***BC_Values,                /* BC values BC_values[wall][variable][m]*/  
            *t_output_vars,             /* array holding output timing info */
                                        /* t_output_vars[1] = t_initial     */
                                        /* t_output_vars[2] = t final       */
                                        /* t_output_vars[3] = delta t       */
            *delt_limits               /* delt_limits[1]   = delt_minimum  */
                                        /* delt_limits[2]   = delt_maximum  */
	;

    int
        **BC_inputs,                    /* BC_types[wall][m] that contains  */
                                        /* the users boundary condition     */
                                        /* selection for each wall          */
        ***BC_types,                    /* each variable can have a Neuman, */
                                        /* or Dirichlet type BC             */
                                        /* BC_types[wall][variable][m]=type */
        ***BC_float_or_fixed           /* BC_float_or_fixed[wall][variable][m]*/
                                        /* Variable on boundary is either   */
                                        /* fixed or it floats during the    */
                                        /* compuation                       */
    ;


    char    output_file_basename[30],   /* Tecplot filename description     */
            output_file_desc[50];       /* Title used in tecplot stuff      */
    int xLoLimit,                       /* x array lower limits             */
        yLoLimit,                       /* y array lower limits             */
        zLoLimit,
        xHiLimit,
        yHiLimit,
        zHiLimit;
     double  delX,                      /* Cell width                       */
             delY,                      /* Cell Width in the y dir          */
             delZ;                      /* Cell width in the z dir          */
    int nMaterials;                     /* Number of materials              */
    ICE(const ICE&);
    ICE& operator=(const ICE&);
};
}
}

#endif
