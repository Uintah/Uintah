
#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H

#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <SCICore/Geometry/Vector.h>

using SCICore::Geometry::Vector;

namespace Uintah {
class ProcessorContext;
class Patch;

namespace ICESpace {

class ICE : public CFDInterface {
public:
   ICE();
   virtual ~ICE();
   
   virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			     SimulationStateP&);
   virtual void scheduleInitialize(const LevelP& level,
				   SchedulerP&,
				   DataWarehouseP&);

   void actuallyInitialize(const ProcessorContext*,
			   const Patch* patch,
			   DataWarehouseP& /* old_dw */,
			   DataWarehouseP& new_dw);
   
   virtual void scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&,
					      DataWarehouseP&);
   void actuallyComputeStableTimestep(const ProcessorContext*,
				      const Patch* patch,
				      DataWarehouseP&,
				      DataWarehouseP&);

   virtual void scheduleTimeAdvance(double t, double dt, 
				    const LevelP&, 
				    SchedulerP&,
				    DataWarehouseP&, 
				    DataWarehouseP&);


   void actuallyTimeStep(const ProcessorContext*,
			 const Patch* patch,
			 DataWarehouseP&,
			 DataWarehouseP&);

   void actuallyStep1(const ProcessorContext*,
			 const Patch* patch,
			 DataWarehouseP&,
		      DataWarehouseP&);

   void actuallyStep2(const ProcessorContext*,
		      const Patch* patch,
		      DataWarehouseP&,
		      DataWarehouseP&);

   void actuallyStep3(const ProcessorContext*,
		      const Patch* patch,
		      DataWarehouseP&,
		      DataWarehouseP&);

   void actuallyStep4(const ProcessorContext*,
		      const Patch* patch,
		      DataWarehouseP&,
		      DataWarehouseP&);

   void actuallyStep5(const ProcessorContext*,
		      const Patch* patch,
		      DataWarehouseP&,
		      DataWarehouseP&);

 void actuallyStep6(const ProcessorContext*,
		    const Patch* patch,
		    DataWarehouseP&,
		    DataWarehouseP&);

 void actuallyStep7(const ProcessorContext*,
		    const Patch* patch,
		    DataWarehouseP&,
		    DataWarehouseP&);


   void convertNR_4dToUCF(const Patch*, CCVariable<Vector>& vel_ucf, 
			  double ****uvel_CC,
			  double ****vvel_CC,
			  double **** wvel_CC,
			  int xLoLimit,
			  int xHiLimit,
			  int yLoLimit,
			  int yHiLimit,
			  int zLoLimit,
			  int zHiLimit,
			  int nMaterials);

 void convertUCFToNR_4d(const Patch*, CCVariable<Vector>& vel_ucf, 
			  double ****uvel_CC,
			  double ****vvel_CC,
			  double **** wvel_CC,
			  int xLoLimit,
			  int xHiLimit,
			  int yLoLimit,
			  int yHiLimit,
			  int zLoLimit,
			  int zHiLimit,
			  int nMaterials);


			  
   
private:
    // These two will go away SOON - a really bad habit, won't work in parallel, blah blah blah
 

    const VarLabel* delTLabel;
    const VarLabel* vel_CCLabel;


   int  i,j,k,m,   
        xLoLimit,                       /* x array lower limits             */
        yLoLimit,                       /* y array lower limits             */
        zLoLimit,
        xHiLimit,
        yHiLimit,
        zHiLimit,
        printSwitch,
        should_I_write_output,          /* flag for dumping output          */
        fileNum,                        /* tecplot file number              */
        stat,                           /* status of putenv and getenv      */
        **BC_inputs,                    /* BC_types[wall][m] that contains  */
                                        /* the users boundary condition     */
                                        /* selection for each wall          */
        ***BC_types,                    /* each variable can have a Neuman, */
                                        /* or Dirichlet type BC             */
                                        /* BC_types[wall][variable][m]=type */
        ***BC_float_or_fixed,           /* BC_float_or_fixed[wall][variable][m]*/
                                        /* Variable on boundary is either   */
                                        /* fixed or it floats during the    */
                                        /* compuation                       */
        nMaterials;                     /* Number of materials              */

                                        
/* ______________________________   
*  Geometry                        
* ______________________________   */     
     double  delX,                      /* Cell width                       */
             delY,                      /* Cell Width in the y dir          */
             delZ,                      /* Cell width in the z dir          */
             delt,
             cheat_t,
             cheat_delt,                /* time step                        */
             CFL,                       /* Courant-Friedrichs and Lewy      */
             t_final,                   /* final problem time               */
            *t_output_vars,             /* array holding output timing info */
                                        /* t_output_vars[1] = t_initial     */
                                        /* t_output_vars[2] = t final       */
                                        /* t_output_vars[3] = delta t       */
            *delt_limits,               /* delt_limits[1]   = delt_minimum  */
                                        /* delt_limits[2]   = delt_maximum  */
                                        /* delt_limits[3]   = delt_initial  */
             t,                         /* current time                     */
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
            ****div_velFC_CC,           /* Divergence of the face centered  */
                                        /* velocity that lives at CC        */    
            ****scalar1_CC,             /* Cell-centered scalars            */   
            ****scalar2_CC,             /* (x, y, z, material)              */
            ****scalar3_CC,
            /*------to be treated as pointers---*/
                                        /*______(x,y,z,face, material)______*/
            ******uvel_FC,              /* u-face-centered velocity         */
            ******vvel_FC,              /* *v-face-centered velocity        */
            ******wvel_FC,              /* w face-centered velocity         */
            ******press_FC,             /* face-centered pressure           */
            ******tau_X_FC,             /* *x-stress component at each face */
            ******tau_Y_FC,             /* *y-stress component at each face */
            ******tau_Z_FC,             /* *z-stress component at each face */
            /*----------------------------------*/                                              
           *grav,                       /* gravity (dir)                    */
                                        /* x-dir = 1, y-dir = 2, z-dir = 3  */
            *gamma,
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
            *R;                         /* gas constant R[material]          */ 
                             
    double  residual,                   /* testing*/            
            x1, x2,
            temp1,
            temp2,
            u0,
            u1,
            u2,
            uL,
            uR;
            
    char    output_file_basename[30],   /* Tecplot filename description     */
            output_file_desc[50];       /* Title used in tecplot stuff      */





    ICE(const ICE&);
    ICE& operator=(const ICE&);
};
}
}

#endif
