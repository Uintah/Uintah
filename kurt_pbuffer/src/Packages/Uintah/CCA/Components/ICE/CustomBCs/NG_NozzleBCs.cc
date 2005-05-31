#define CONVERGENCE 1e-6
#define MAX_ITER 50 

#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/NG_NozzleBCs.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include <iostream>

using namespace Uintah;
using namespace SCIRun;

static DebugStream cout_doing("NG_DOING_COUT", false);
static DebugStream cout_dbg("NG_DBG_COUT",     false);
/*______________________________________________________________________
Our goal is to come up with reasonable values for the pressure
temperature, density and velocity in the ghost cell (see fig 1).  The method
specified by Northrup Grumman was to use a solution to the 1D shock
tube problem at every nozzle inlet cell and at every timestep.

First, define the location of the stagnation conditions to
be 1 delta X away from the ghostcell cell-center (see Fig 1)
You could define it to be 5 or even 10 delta X away.  This
is an imaginary location outside our our computational domain.  
Our grid stops at point (g).


  One row of cells in the nozzle      
        ----------------------------  o = location of stagnation conditions          
       |      |      |       |            (pressure, temperature, density known)
   o   |  g   |  i   |  i    |   i    g = ghostcell
       |      |      |       |
        ----------------------------  i = interior cells                             
   ^______^                               (state vector known)                      
    deltaX            Fig 1


Now define a secondary set of points over which the shock tube (S.T.) solution
 will be computed.  At each point (.) the solution will provide
the static density, temperature, pressure and velocity.

           ------------------------------ 
          |            |             |
          |            |             |    . = points for shock tube problem
   o......|.....g......|.......i     |
          |            |             |
          |            |             |
           ------------------------------
                       ^
                (diaphragm location is assumed to be here)
              
                     Fig 2
                    
                    
Here is some pseudo code showing the algorithm.
__________________________________
while ( simulation time  < 43 msec) do
    
    < Advance the solution on the interior grid cells>
    
    Set boundary conditions()
      Loop( all nozzle ghost cells) do 
   
        - Compute shock tube problem()
  
        - Pick off the pressure, temperature and velocity at 
          point (g) from the ST solution.
    
        - Insert pressure, temperature, density and 
          velocity into the ghostcell cell-centers
    end Loop
  end setBoundaryConditions
end while
__________________________________

The right state, at point i is simply the state vector at one cell in.
______________________________________________________________________*/

#define NR_END 1
#define FREE_ARG char*

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

double *dvector_nr(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector_nr()");
	return v-nl+NR_END;
}

void free_dvector_nr(double *v, long nl, long /*nh*/)
/* free a double vector allocated with dvector_nr() */
{
	free((FREE_ARG) (v+nl-NR_END));
}
//______________________________________________________________________
//  compute Stagnation properties from curve fits
void
Uintah::computeStagnationProperties(double &stag_press,
                                    double &stag_temp,
                                    double &stag_rho,
                                    double &time,
                                    SimulationStateP& sharedState)
{
    //__________________________________
    // constants
    double pascals_per_psi = 6894.4;
    double gamma = 1.4;
    double cv    = 716;
    //__________________________________
    //  curve fit for the pressure
    double mean = 3.2e-02;
    double std  = 1.851850425925377e-02;
    vector<double> c(8);
    c[7] = -2.233767893099373e+05;
    c[6] = -5.824923861605825e+04;
    c[5] =  1.529343908400654e+06;
    c[4] =  1.973796762592652e+05;
    c[3] =  -3.767381404747613e+06;
    c[2] =  5.939587204102841e+04;
    c[1] =  5.692124635957888e+06;    
    c[0] =  3.388928006241853e+06;

    time = sharedState->getElapsedTime();
    //double time = t + sharedState->getTimeOffset();

    double tbar = (time - mean)/std;

    stag_press = c[7] * pow(tbar,7) + c[6]*pow(tbar,6) + c[5]*pow(tbar,5)
               + c[4] * pow(tbar,4) + c[3]*pow(tbar,3) + c[2]*pow(tbar,2)
               + c[1] * tbar + c[0];
              
    stag_temp  = (2.7988e3 +110.5*log(stag_press/pascals_per_psi) ); 
    
    stag_rho   = stag_press/((gamma - 1.0) * cv * stag_temp); 
}

/*______________________________________________________________________
 Function: BC_values_using_IsentropicRelations
 Compute the static (temperature, density, velocity)  using isentropic
 relations (gamma = 1.4) and an area ratio = 1.88476 
______________________________________________________________________*/
void
Uintah::BC_values_using_IsentropicRelations(const double stag_press,
                                            const double stag_rho,
                                            const double stag_temp,
                                            double &static_press,
                                            double &static_temp,
                                            double &static_rho,
                                            double &static_vel)
{
  //__________________________________
  //  Isenentrop relations for A/A* = 1.88476
  double p_p0     = 0.92850;
  double T_T0     = 0.97902;
  double rho_rho0 = 0.94839;
  double Mach     = 0.32725;
  double R        = 287.0;
  double gamma    = 1.4;
  static_temp  = T_T0 * stag_temp;
  static_rho   = rho_rho0 * stag_rho;
  static_press = p_p0 * stag_press;
  static_vel   = Mach * sqrt(gamma * R * static_temp); 
}

/*______________________________________________________________________
 Function:  p2_p1_ratio--MISC: computes p2/p1           
 This is function is used by the secant method.
______________________________________________________________________*/
double
Uintah::p2_p1_ratio( double  gamma,
                     double  p4_p1,
                     double  p2_p1_guess,
                     double  a4,
                     double  a1,
                     double  u4,
                     double  u1 )
{         
  double gamma_ratio1, gamma_ratio2,
         sqroot, exponent, fraction, boxed_quantity;
         
  gamma_ratio1    = (gamma + 1.0)/( 2.0 * gamma);
  sqroot          = sqrt( gamma_ratio1 * (p2_p1_guess - 1.0) + 1.0 );
  fraction        = (p2_p1_guess - 1.0)/sqroot;
  
  boxed_quantity  = u4 - u1 - (a1/gamma) * fraction;

  gamma_ratio2    = (gamma - 1.0)/(2.0 * a4);
  exponent        = -2.0*gamma/(gamma - 1.0);
  double p2_p1;
  p2_p1 =  p4_p1 - p2_p1_guess * pow( (1.0 + gamma_ratio2 * boxed_quantity), exponent);
  //_________________________
  // Bulletproofing
  if (finite(p2_p1) == 0) {
    cout << " p4_p1 " << p4_p1 << " p2_p1_guess " << p2_p1_guess
         << " boxed_quantity "<< boxed_quantity
         << " exponent " << exponent
         << " sqrt " << sqroot
         << " fraction " << fraction << endl;
    cout << " boxed_quantity: u4 " << u4 << " u1 " << u1 << " a1 " << a1 << " gamma " << gamma
         << " fraction " << fraction << endl;
    cout << " (1.0 + gamma_ratio2 * boxed_quantity) " << (1.0 + gamma_ratio2 * boxed_quantity) << endl;
    throw InternalError( "p2_p1_ratio: I've computed a nan or inf for p2_p1");
  }
  return p4_p1 - p2_p1_guess * pow( (1.0 + gamma_ratio2 * boxed_quantity), exponent);
}


/*______________________________________________________________________
 Function:  Solve_Riemann_problem--
 Purpose:   Solves the shock tube problem
   
Implementation Notes:
    The notation comes from the reference
       
Reference:
    Computational GasDynamics by C.B. Laney, 1998, pg 73
______________________________________________________________________*/
void
Uintah::Solve_Riemann_problem(
        int     qLoLimit,                /* array lower limit                */
        int     qHiLimit,                /* upper limit                      */
        double  diaphragm_location,      /*diaphram location                 */
        double  delQ,
        double  time,
        double  gamma,
        double  p1,                     /* pressure  right of the diaphram  */
        double  rho1,                   /* density                          */
        double  u1,                     /* velocity                         */
        double  a1,
        double  p4,                     /* pressure  Left of the diaphram   */
        double  rho4,                   /* density                          */
        double  u4,                     /* velocity                         */ 
        double  a4,
        double  *u_Rieman,              /* exact solution velocity          */
        double  *a_Rieman,              /* exact solution speed of sound    */
        double  *p_Rieman,              /* exact solution pressure          */
        double  *rho_Rieman,            /* exact solution density           */  
        double  *T_Rieman,
        double  R)                      /* ideal gas constant               */      
{
    int     i, iter;
            
    double  gamma_ratio1,               /* variables that make writing the eqs easy */
            gamma_ratio2,
            exponent,
            sqroot,
            fraction;
    double 
            S,                          /* shock velocity                   */  
            x,       
            xtemp,    
            xshock,                     /* location of the shock            */        
            xcontact,                   /* location of the contact          */      
            xexpansion_head,            /* Location of the expansion head   */
            xexpansion_tail;            /* location of the expansion tail   */

    double  p2, p3,                     /* quantities in the various locations*/
            u2, u3,
            a2, a3,
            rho2, rho3,
            p2_p1,
            p4_p1;
            
    double  p2_p1_guess_old,
            p2_p1_guess_new,
            p2_p1_guess0,
            p2_p1_guess00;
            
    double  delta;                       /* used by the secant method        */
          //fudge;        
    static double p2_p1_guess;
/*__________________________________
*   Compute some stuff
*___________________________________*/
    p4_p1           = p4/p1;
    if (p2_p1_guess > 1){
      p2_p1_guess0    = 0.9* p2_p1_guess;
      p2_p1_guess00   = 1.1* p2_p1_guess;
    }else{
      p2_p1_guess0    = 0.5 * p4_p1;
      p2_p1_guess00   = 2.0 * p4_p1;
    }
    iter              = 0;
/*______________________________________________________________________
*   Use the secant method to solve for pressure ratio across the shock
*                   p2_p1
*   See Numerical Methods by Hornbeck, pg 71
*_______________________________________________________________________*/
/*__________________________________
*   Step 1 Compute the pressure ratio
*   across the shock
*   Need to add iterative loop
*___________________________________*/
    delta               = p2_p1_guess0 - p2_p1_guess00;
    p2_p1_guess         = p2_p1_guess0;
    p2_p1_guess_old     = p2_p1_ratio( gamma,  p4_p1, p2_p1_guess00, a4, a1, u4, u1  );

    while (fabs(delta) > CONVERGENCE && iter < MAX_ITER){
      p2_p1_guess_new = p2_p1_ratio( gamma,  p4_p1, p2_p1_guess, a4, a1, u4, u1  );
      delta           = -(p2_p1_guess_new * delta )/(p2_p1_guess_new - p2_p1_guess_old + 1e-100);
      p2_p1_guess     = p2_p1_guess + delta;

      p2_p1_guess_old = p2_p1_guess_new;

      iter ++;
    }
    p2_p1               = p2_p1_guess;
    
    //____________________
    // bullet proofing
    if(finite(p2_p1) == 0){
      cout << " ---------------------------ERROR" << endl;
      cout << " p4/p1 " << p4_p1 << " p4 " << p4 << " p1 " << p1 << endl;
      cout << " p2_p1_guess0 " << p2_p1_guess0 << " p2_p1_guess00 " << p2_p1_guess00 << endl;
      cout << " p2_p1 " << p2_p1 << endl;
      throw InternalError("Solve_Riemann_problem: p2_p1 is either inf or nan");
    }
/*______________________________________________________________________
*   Now compute the properties
*   that are constant in each section
*_______________________________________________________________________*/
    gamma_ratio1        = (gamma + 1.0)/( 2.0 * gamma);
    sqroot              = sqrt( gamma_ratio1 * (p2_p1 - 1.0) + 1.0 );
    fraction            = (p2_p1 - 1.0)/sqroot;
    u2                  = u1 + (a1/gamma) * fraction;
 
    gamma_ratio1        = (gamma + 1.0)/(gamma - 1.0);
    fraction            = (gamma_ratio1 + p2_p1)/( 1.0 + gamma_ratio1 * p2_p1);
    a2                  = sqrt(pow(a1,2.0) * p2_p1 * fraction );
    p2                  = p2_p1 * p1;
    rho2                = gamma * p2/pow(a2,2.0);
 
    /*___*/
    u3                  = u2;
    p3                  = p2;
    
    gamma_ratio1        = (gamma -1.0)/2.0;
    a3                  = gamma_ratio1*(u4 + a4/gamma_ratio1 - u3);
    rho3                = gamma * p3/pow(a3,2.0);
   
/*______________________________________________________________________
*   Step 2 Compute the shock and expansion locations all relative the orgin
*   Write the data to an array
*_______________________________________________________________________*/
    gamma_ratio1        = (gamma + 1.0)/( 2.0 * gamma);
    sqroot              = sqrt(gamma_ratio1 * (p2_p1 - 1.0) + 1.0);
    S                   = u1 + a1 * sqroot;
    
    xshock              = diaphragm_location + (S  * time);             
    xcontact            = diaphragm_location + (u3 * time);             
    xexpansion_head     = diaphragm_location + ((u4 - a4)  * time);
    xexpansion_tail     = diaphragm_location + ((u3 - a3)  * time);   
    
     //__________________________________
     //  compute common reused variables
     double gamma_R     = (gamma * R);
     double a1_sqr      = a1 * a1;
     double a2_sqr      = a2 * a2;
     double a3_sqr      = a3 * a3;
        
    //__________________________________
    //   Now write all of the data to the arrays

    for( i = qLoLimit; i <= qHiLimit; i++){
      x = (double) (i-qLoLimit) * delQ;
     //__________________________________
     //  Region 1
      if (x >=xshock){
        u_Rieman[i]     = u1; 
        a_Rieman[i]     = a1;
        p_Rieman[i]     = p1;
        rho_Rieman[i]   = rho1;
        T_Rieman[i]     = a1_sqr/gamma_R;
      }
      //__________________________________
      //   Region 2
      if ( (xcontact < x) && (x < xshock) ){
        u_Rieman[i]     = u2; 
        a_Rieman[i]     = a2;
        p_Rieman[i]     = p2;
        rho_Rieman[i]   = rho2;
        T_Rieman[i]     = a2_sqr/gamma_R;
      }
      //__________________________________
      //   Region 3
      if ( (xexpansion_tail <= x) && (x <= xcontact) ){
        u_Rieman[i]     = u3; 
        a_Rieman[i]     = a3;
        p_Rieman[i]     = p3;
        rho_Rieman[i]   = rho3;
        T_Rieman[i]     = a3_sqr/gamma_R;
      }
      //__________________________________
      //   Expansion fan Between 3 and 4
      if ( (xexpansion_head <= x) && (x < xexpansion_tail) ){
        xtemp           = (x - xexpansion_tail);
        exponent        = (2.0 * gamma)/( gamma - 1.0 );
        gamma_ratio2    = 2.0/(gamma + 1.0);
        u_Rieman[i]     =  gamma_ratio2 * ( xtemp/(time + 1.0e-100) + ((gamma - 1.0)/2.0) * u4 + a4);

        a_Rieman[i]     =  u_Rieman[i] - xtemp/(time + 1.0e-100);
        p_Rieman[i]     =  p4 * pow( (a_Rieman[i]/a4), exponent);

        exponent        = (2.0)/( gamma - 1.0 );
        rho_Rieman[i]   =  rho4 * pow( (a_Rieman[i]/a4), exponent);

        T_Rieman[i]     =  a_Rieman[i]*a_Rieman[i]/(gamma * R);
      }
      //__________________________________
      //   Region 4
      if (x <xexpansion_head){
        u_Rieman[i]     = u4;
        a_Rieman[i]     = a4;
        p_Rieman[i]     = p4;
        rho_Rieman[i]   = rho4;
        T_Rieman[i]     = a4*a4/(gamma*R);
      }
    }

#if 0
    fprintf(stderr,"________________________________________________\n"); 
    fprintf(stderr," p2_p1: %f\n",p2_p1);
    fprintf(stderr," u4: %f,\t  u3: %f,\t  u2: %f,\t  u1: %f\n",u4, u3, u2, u1);
    fprintf(stderr," a4: %f,\t  a3: %f,\t  a2: %f,\t  a1: %f\n",a4, a3, a2, a1);
    fprintf(stderr," p4: %f,\t  p3: %f,\t  p2: %f,\t  p1: %f\n",p4, p3, p2, p1); 
    fprintf(stderr," rho4: %f,\t  rho3: %f,\t  rho2: %f,\t  rho1: %f\n",rho4, rho3, rho2, rho1); 
    fprintf(stderr," shock Velocity: \t %f \n",S);
    fprintf(stderr," LHS expansion vel: \t %f \n",(u4 - a4) );
    fprintf(stderr," RHS expansion vel: \t %f \n",(u3 - a3) );    
    fprintf(stderr," Xlocations\n");
    fprintf(stderr,"%f,  %f,  %f,   %f\n",xexpansion_head, xexpansion_tail, xcontact, xshock);
    fprintf(stderr,"________________________________________________\n");   
#endif    
}
//______________________________________________________________________
void
Uintah::solveRiemannProblemInterface( const double t_final,
                                      const double Length, int ncells,
                                      const double u4, const double p4, const double rho4,
                                      const double u1, const double p1, const double rho1,
                                      const double diaphragm_location,
                                      const int probeCell,
                                      NG_BC_vars* ng,
                                      double &press,    // at probe location
                                      double &Temp,
                                      double &rho,
                                      double &vel)
{
  int    qLoLimit, qHiLimit, Q_MAX_LIM,i;                                         
  double a1,     a4,
         *u_Rieman, *a_Rieman,*p_Rieman,*rho_Rieman,*T_Rieman,
         R, gamma, delQ;
  //__________________________________
  //Parse arguments                
//  cout_dbg << " p4 " << p4 << " u4 " << u4 << " rho4 " << rho4
//           << " p1 " << p1 << " u1 " << u1 << " rho1 " << rho1 << endl;
//  cout_dbg << " t_final " << t_final<< " length " << Length << " resolution " << ncells
//           << " diaphragm loc: " << diaphragm_location << endl;
  //__________________________________
  //  allocate memory
  Q_MAX_LIM   = 505;
  u_Rieman    = dvector_nr(0, Q_MAX_LIM);
  a_Rieman    = dvector_nr(0, Q_MAX_LIM);
  p_Rieman    = dvector_nr(0, Q_MAX_LIM);
  rho_Rieman  = dvector_nr(0, Q_MAX_LIM);
  T_Rieman    = dvector_nr(0, Q_MAX_LIM);
  //__________________________________
  //  initialize variables 
  qLoLimit = 1;
  qHiLimit = ncells;
  delQ     = (double)Length/ncells;
  if(qHiLimit >= Q_MAX_LIM) {
    fprintf(stderr, "ERROR: qHiLimit > Q_MAX_LIM");
    exit(1);
  }                                                           
  gamma    = 1.4;                                                             
  R        = 287;                                                              
  a4       = sqrt(gamma * p4 /rho4);                                         
  a1       = sqrt(gamma * p1 /rho1);                                         

  //__________________________________
  // NOTE: this analysis assumes that an ideal gas with identical properties
  //         is used in both partitions.
  for ( i = qLoLimit; i <= qHiLimit; i++){
    u_Rieman[i]     = -9.0;
    a_Rieman[i]     = -9.0;
    p_Rieman[i]     = -9.0;
    rho_Rieman[i]   = -9.0; 
    T_Rieman[i]     = -9.0;          
  }
          
  Solve_Riemann_problem(  qLoLimit,       qHiLimit,       diaphragm_location ,
                          delQ,           t_final,        gamma,
                          p1,             rho1,           u1, a1,
                          p4,             rho4,           u4, a4,
                          u_Rieman,       a_Rieman,       p_Rieman,
                          rho_Rieman,     T_Rieman,       R);  

  //__________________________________
  // find values at the probe location
  press = p_Rieman[probeCell];
  rho   = rho_Rieman[probeCell];
  vel   = u_Rieman[probeCell];
  Temp  = T_Rieman[probeCell];    
  //__________________________________
  // dump to a file when sus dumps and at
  //  cell index -1, 1, 0  
          
  if(ng->dumpNow){
    if(ng->dataArchiver->wasOutputTimestep() && ng->c == IntVector(-1,1,0)) {
    
      string udaDir    = ng->dataArchiver->getOutputLocation();
      
      DIR *check = opendir(udaDir.c_str());
      if ( check != NULL){        // only dump if uda directory exists
      
        ostringstream dw;
        dw << ng->dataArchiver->getCurrentTimestep(); 
        string filename = udaDir + "/exactSolution_" + dw.str();

        double x = 0;
        cout << " Dumping file " << filename  << endl;
         FILE *fp = fopen(filename.c_str(),"w");
         /*fprintf(fp, "#X p_Rieman u_Rieman rho_Rieman a_Rieman T_Rieman sp_vol_Rieman\n");*/
         for ( i = qLoLimit; i <= qHiLimit; i++){
           fprintf(fp, "%6.5E %6.5E %6.5E %6.5E %6.5E %6.5E %6.5E\n", 
                   x, p_Rieman[i], u_Rieman[i], rho_Rieman[i], 
                   a_Rieman[i], T_Rieman[i], 1.0/rho_Rieman[i]);
           x +=delQ;
         }
         fclose(fp);
        }
    }
  } 
  //__________________________________
  //   free memory
  //fprintf(stderr,"Now deallocating memory\n"); 
  free_dvector_nr( u_Rieman,     0, Q_MAX_LIM);
  free_dvector_nr( a_Rieman,     0, Q_MAX_LIM);
  free_dvector_nr( p_Rieman,     0, Q_MAX_LIM);
  free_dvector_nr( rho_Rieman,   0, Q_MAX_LIM); 
  free_dvector_nr( T_Rieman,     0, Q_MAX_LIM); 
}

//______________________________________________________________________
void
Uintah::setNGCVelocity_BC(const Patch* patch,
                          const Patch::FaceType face,
                          CCVariable<Vector>& q_CC,
                          const string& var_desc,
                          const vector<IntVector> bound,
                          const string& bc_kind,
			     const int mat_id,
			     const int child,
                          SimulationStateP& sharedState,
                          NG_BC_vars* ng)
{
  if (var_desc == "Velocity" && bc_kind == "Custom") {
    cout_doing << "Doing setNGCVelocity_BC " << patch->getID() << endl;
    
    setNGC_Nozzle_BC<CCVariable<Vector>, Vector>
          (patch, face, q_CC, var_desc,"CC", bound, bc_kind,mat_id, 
           child, sharedState, ng);

    // set the y and z velocity components to 0
    vector<IntVector>::const_iterator iter;
    for (iter=bound.begin(); iter != bound.end(); iter++) {
      IntVector c = *iter;
      q_CC[c].y(0.0);
      q_CC[c].z(0.0);
    }
  } 
}

//______________________________________________________________________
// add the requires needed by each of the various tasks
void
Uintah::addRequires_NGNozzle(Task* t, 
                             const string& where,
                             ICELabel* lb,
                             const MaterialSubset* ice_matls)
{
  cout_doing<< "Doing addRequires_NGNozzle: \t\t" <<t->getName()
            << " " << where << endl;
  
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();
  
  
  if(where == "EqPress"){
    t->requires(Task::OldDW, lb->vel_CCLabel, ice_matls,  gn,0);
    t->requires(Task::OldDW, lb->rho_CCLabel, ice_matls,  gn,0);
  }
  if(where == "velFC_Exchange"){
    t->requires(Task::OldDW, lb->rho_CCLabel, ice_matls,   gn,0);        
    t->requires(Task::OldDW, lb->vel_CCLabel, ice_matls,   gn,0);        
    t->requires(Task::NewDW, lb->press_equil_CCLabel, press_matl, oims, gn);
  }
  if(where == "imp_velFC_Exchange"){
    t->requires(Task::ParentOldDW, lb->rho_CCLabel, ice_matls,   gn,0);        
    t->requires(Task::ParentOldDW, lb->vel_CCLabel, ice_matls,   gn,0);        
    t->requires(Task::ParentNewDW, lb->press_equil_CCLabel, press_matl, oims, gn);
  }
  if(where == "update_press_CC"){
    t->requires(Task::OldDW, lb->vel_CCLabel, ice_matls, gn,0);
  }
  if(where == "imp_update_press_CC"){
    t->requires(Task::ParentOldDW, lb->vel_CCLabel, ice_matls, gn,0);
  }
  if(where == "CC_Exchange"){
    t->requires(Task::NewDW, lb->rho_CCLabel,   ice_matls,   gn,0);
    t->requires(Task::NewDW, lb->press_CCLabel, press_matl, oims, gn);
    t->computes(lb->vel_CC_XchangeLabel);
    t->computes(lb->temp_CC_XchangeLabel);
  }
  if(where == "Advection"){
    t->requires(Task::NewDW, lb->press_CCLabel, press_matl, oims, gn);
  }
}

//______________________________________________________________________
// get the necessary data an push it into the NG_vars struct
void
Uintah::getVars_for_NGNozzle( DataWarehouse* old_dw,
                              DataWarehouse* new_dw,
                              ICELabel* lb,
                              const Patch* patch,
                              const string& where,
                              bool& setNG_Bcs,
                              NG_BC_vars* ng)
{

  cout_doing <<  "Doing getVars_for_NGNozzle Patch:" << patch->getID() 
             << " " << where << endl;
  Ghost::GhostType  gn  = Ghost::None;
  int hardCodedIndx = 1;  // index of ICE matl   
  ng->dumpNow = false;    // dump out gnuplot of boundary condition
  setNG_Bcs   = true;     // set boundary conditions in every task.
  
  new_dw->allocateTemporary(ng->press_CC, patch);
  new_dw->allocateTemporary(ng->rho_CC,   patch);
  new_dw->allocateTemporary(ng->vel_CC,   patch);
  
  //__________________________________
  //    EqPress
  if(where == "EqPress"){
    int hardCodedIndx = 1;
    constCCVariable<Vector> vel;
    old_dw->get(vel, lb->vel_CCLabel, hardCodedIndx,patch,gn,0);
       
    ng->vel_CC.copyData(vel);
  }
  if(where == "EqPressMPMICE"){
    constCCVariable<Vector> vel;
    CCVariable<double> press, rho;
    new_dw->getCopy(press, lb->press_equil_CCLabel, 0,      patch,gn,0);
    new_dw->getCopy(rho,   lb->rho_CCLabel,   hardCodedIndx,patch,gn,0);
    old_dw->get(vel,       lb->vel_CCLabel,   hardCodedIndx,patch,gn,0);

    ng->press_CC.copyData(press);
    ng->vel_CC.copyData(vel);
    ng->rho_CC.copyData(rho);
  }
  
  //__________________________________
  //    FC exchange
  if(where == "velFC_Exchange"){
    constCCVariable<double> rho, press;
    constCCVariable<Vector> vel;
    
    old_dw->get(rho,      lb->rho_CCLabel,    hardCodedIndx,patch,gn,0);
    old_dw->get(vel,      lb->vel_CCLabel,    hardCodedIndx,patch,gn,0);
    new_dw->get(press,    lb->press_equil_CCLabel, 0,       patch,gn,0);
    
    ng->press_CC.copyData(press);
    ng->rho_CC.copyData(rho);
    ng->vel_CC.copyData(vel);
  }
  
  //__________________________________
  //    update pressure
  if(where == "update_press_CC"){
    constCCVariable<Vector> vel;
    constCCVariable<double> rho;
    CCVariable<double> press;
    new_dw->getCopy(press, lb->press_CCLabel, 0,          patch,gn,0);
    old_dw->get(vel,       lb->vel_CCLabel, hardCodedIndx,patch,gn,0);     
    old_dw->get(rho,       lb->rho_CCLabel, hardCodedIndx,patch,gn,0);  
    
    ng->press_CC.copyData(press);
    ng->vel_CC.copyData(vel);
    ng->rho_CC.copyData(rho); 
  }
  //__________________________________
  //    cc_ Exchange
  if(where == "CC_Exchange"){
    constCCVariable<double> press, rho;
    constCCVariable<Vector> vel_CC;
    old_dw->get(rho,     lb->rho_CCLabel,         hardCodedIndx,patch,gn,0);    
    new_dw->get(press,   lb->press_CCLabel,       0,            patch,gn,0);
    new_dw->get(vel_CC,  lb->vel_CC_XchangeLabel, hardCodedIndx,patch,gn,0); 
        
    ng->press_CC.copyData(press);
    ng->rho_CC.copyData(rho);
    ng->vel_CC.copyData(vel_CC);
  cout << " done with get_Vars for NG " << endl;
  }
  //__________________________________
  //    Advection
  if(where == "Advection"){
    constCCVariable<double> press;
    CCVariable<double> rho;
    CCVariable<Vector> vel;
    
    new_dw->getCopy(rho, lb->rho_CCLabel,   hardCodedIndx, patch,gn,0);
    new_dw->getCopy(vel, lb->vel_CCLabel,   hardCodedIndx, patch,gn,0);
    new_dw->get(press,   lb->press_CCLabel, 0,             patch,gn,0);
    
    ng->press_CC.copyData(press);
    ng->rho_CC.copy(rho);
    ng->vel_CC.copy(vel);
    ng->dumpNow = true;
  }
}

/* ______________________________________________________________________
 Function~  using_NG_hack
 Purpose~   returns if we are using the Northrup Grumman BC hack on any face,
 ______________________________________________________________________  */
bool
Uintah::using_NG_hack(const ProblemSpecP& prob_spec)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if custom bcs are specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
 
  bool usingNG = false;
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
    map<string,string> face;
    face_ps->getAttributes(face);
        
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      

      if (bc_type["var"] == "Custom") {
        usingNG = true;
      }
    }
  }
  return usingNG;
}





