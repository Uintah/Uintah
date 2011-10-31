//----- Ray.cc ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Components/Models/Radiation/RMCRT/MersenneTwister.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/DbgOutput.h>
#include <time.h>

//--------------------------------------------------------------
//
using namespace Uintah;
using namespace std;
static DebugStream dbg("RAY", false);
//---------------------------------------------------------------------------
// Method: Constructor. he's not creating an instance to the class yet
//---------------------------------------------------------------------------
Ray::Ray()
{
  _pi = acos(-1); 

  d_sigmaT4_label = VarLabel::create( "sigmaT4", CCVariable<double>::getTypeDescription() ); 
  d_matlSet = 0;
}

//---------------------------------------------------------------------------
// Method: Destructor
//---------------------------------------------------------------------------
Ray::~Ray()
{
  VarLabel::destroy(d_sigmaT4_label);
  
  if(d_matlSet && d_matlSet->removeReference()) {
    delete d_matlSet;
  }
}

//---------------------------------------------------------------------------
// Method: Problem setup (access to input file information)
//---------------------------------------------------------------------------
void
Ray::problemSetup( const ProblemSpecP& inputdb) 
{
  ProblemSpecP db = inputdb;

  db->getWithDefault( "NoOfRays"  , _NoOfRays  , 1000 );
  db->getWithDefault( "Threshold" , _Threshold , 0.01 );      //When to terminate a ray
  db->getWithDefault( "Alpha"     , _alpha     , 0.2 );       //Absorption coefficient of the boundaries
  db->getWithDefault( "Slice"     , _slice     , 9 );         //Level in z direction of xy slice
  db->getWithDefault( "benchmark_1" , _benchmark_1, false );  //probably need to make this smarter...
                                                              //depending on what isaac has in mind
  db->getWithDefault("StefanBoltzmann", _sigma, 5.67051e-8);  // Units are W/(m^2-K)
  _sigma_over_pi = _sigma/_pi;

}

//______________________________________________________________________
// Register the material index and label names
void
Ray::registerVarLabels(int   matlIndex,
                       const VarLabel* abskg,
                       const VarLabel* absorp,
                       const VarLabel* temperature,
                       const VarLabel* divQ)
{
  d_matl             = matlIndex;
  d_abskgLabel       = abskg;
  d_absorpLabel      = absorp;
  d_temperatureLabel = temperature;
  d_divQLabel        = divQ;
  
  //__________________________________
  //  define the materialSet
  d_matlSet = scinew MaterialSet();
  vector<int> m;
  m.push_back(matlIndex);
  d_matlSet->addAll(m);
  d_matlSet->addReference();
}
//---------------------------------------------------------------------------
//
void 
Ray::sched_initProperties( const LevelP& level, SchedulerP& sched, const int time_sub_step )
{

  std::string taskname = "Ray::schedule_initProperties"; 
  Task* tsk = scinew Task( taskname, this, &Ray::initProperties, time_sub_step ); 
  printSchedule(level,dbg,taskname);
  
  if ( time_sub_step == 0 ) { 
    tsk->requires( Task::OldDW, d_abskgLabel,       Ghost::None, 0 ); 
    tsk->requires( Task::OldDW, d_temperatureLabel, Ghost::None, 0 ); 
    tsk->computes( d_sigmaT4_label ); 
    tsk->computes( d_abskgLabel ); 
    tsk->computes( d_absorpLabel );

  } else { 
    tsk->requires( Task::NewDW, d_temperatureLabel, Ghost::None, 0 ); 
    tsk->modifies( d_sigmaT4_label ); 
    tsk->modifies( d_abskgLabel ); 
    tsk->modifies( d_absorpLabel ); 
  }

  sched->addTask( tsk, level->eachPatch(), d_matlSet ); 

}
//______________________________________________________________________
//
void
Ray::initProperties( const ProcessorGroup* pc,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw, 
                     const int time_sub_step )
{
  // patch loop
  const Level* level = getLevel(patches);
  
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::InitProperties");
    
    CCVariable<double> abskg; 
    CCVariable<double> absorp; 
    CCVariable<double> sigmaT4;

    constCCVariable<double> temperature; 

    if ( time_sub_step == 0 ) { 

      new_dw->allocateAndPut( abskg,    d_abskgLabel,     d_matl, patch ); 
      new_dw->allocateAndPut( sigmaT4,  d_sigmaT4_label,  d_matl, patch ); 
      new_dw->allocateAndPut( absorp,   d_absorpLabel,    d_matl, patch ); 

      abskg.initialize  ( 0.0 ); 
      absorp.initialize ( 0.0 ); 
      sigmaT4.initialize( 0.0 );

      old_dw->get(temperature,      d_temperatureLabel, d_matl, patch, Ghost::None, 0);

    } else { 

      new_dw->getModifiable( sigmaT4, d_sigmaT4_label,  d_matl, patch ); 
      new_dw->getModifiable( absorp,  d_absorpLabel,    d_matl, patch ); 
      new_dw->getModifiable( abskg,   d_abskgLabel,     d_matl, patch ); 
      new_dw->get( temperature,     d_temperatureLabel, d_matl, patch, Ghost::None, 0 ); 

    }
    
    IntVector pLow;
    IntVector pHigh;
    level->findInteriorCellIndexRange(pLow, pHigh);

    int Nx = pHigh[0] - pLow[0];
    int Ny = pHigh[1] - pLow[1];
    int Nz = pHigh[2] - pLow[2];   

    Vector Dx = patch->dCell(); 

    for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){ 

      IntVector c = *iter; 

      if ( _benchmark_1 ) { 
        abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( ( c[0] - (Nx - 1.0) /2.0) * Dx[0]) )
                        * ( 1.0 - 2.0 * fabs( ( c[1] - (Ny - 1.0) /2.0) * Dx[1]) )
                        * ( 1.0 - 2.0 * fabs( ( c[2] - (Nz - 1.0) /2.0) * Dx[2]) ) 
                        + 0.1;

      } else { 

        // need to put radcal calulation here: 
        abskg[c] = 0.0; 
        absorp[c] = 0.0; 

      } 
      double temp2 = temperature[c] * temperature[c] ;
      sigmaT4[c] = _sigma_over_pi * temp2 * temp2; // \sigma T^4

    }
  }
}


//---------------------------------------------------------------------------
// 
//---------------------------------------------------------------------------
  void
Ray::sched_sigmaT4( const LevelP& level, 
                    SchedulerP& sched )
{

  std::string taskname = "Ray::sched_sigmaT4";
  Task* tsk= scinew Task( taskname, this, &Ray::sigmaT4 );

  printSchedule(level,dbg,taskname);
  
  tsk->requires( Task::OldDW, d_temperatureLabel, Ghost::None, 0 ); 
  tsk->computes(d_sigmaT4_label); 

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//---------------------------------------------------------------------------
// Compute total intensity over all wave lengths (sigma * Temperature^4)
//---------------------------------------------------------------------------
void
Ray::sigmaT4( const ProcessorGroup*,
              const PatchSubset* patches,           
              const MaterialSubset*,                
              DataWarehouse* old_dw,                
              DataWarehouse* new_dw )               
{

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::sigmaT4");
   
    double sigma_over_pi = _sigma/M_PI;
    
    constCCVariable<double> temp;
    constCCVariable<double> abskg;
    CCVariable<double> sigmaT4;             // sigma T ^4/pi
    
    old_dw->get(temp,               d_temperatureLabel,   d_matl, patch, Ghost::None, 0);  
    new_dw->allocateAndPut(sigmaT4, d_sigmaT4_label,      d_matl, patch);

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      double T_sqrd = temp[c] * temp[c];
      sigmaT4[c] = sigma_over_pi * T_sqrd * T_sqrd;
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the ray tracer
//---------------------------------------------------------------------------
void
Ray::sched_rayTrace( const LevelP& level, SchedulerP& sched, const int time_sub_step )
{
  std::string taskname = "Ray::sched_rayTrace";
  Task* tsk= scinew Task( taskname, this, &Ray::rayTrace, time_sub_step );
  printSchedule(level,dbg,taskname);

  // require an infinite number of ghost cells so  you can access
  // the entire domain.
  Ghost::GhostType  gac  = Ghost::AroundCells;
  tsk->requires( Task::NewDW , d_abskgLabel  ,  gac, SHRT_MAX);
  tsk->requires( Task::NewDW , d_sigmaT4_label, gac, SHRT_MAX);
//  tsk->requires( Task::OldDW , d_lab->d_cellTypeLabel , Ghost::None , 0 );

  if( time_sub_step == 0 ){
    tsk->computes( d_divQLabel ); 
  } else {
    tsk->modifies( d_divQLabel );
  }
  sched->addTask( tsk, level->eachPatch(), d_matlSet );

}

//---------------------------------------------------------------------------
// Method: The actual work of the ray tracer
//---------------------------------------------------------------------------
void
Ray::rayTrace( const ProcessorGroup* pc,
               const PatchSubset* patches,
               const MaterialSubset* matls,
               DataWarehouse* old_dw,
               DataWarehouse* new_dw,
               const int time_sub_step )
{ 
  const Level* level = getLevel(patches);
  int maxLevels = level->getGrid()->numLevels();
  MTRand _mTwister; 
  
  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  level->findInteriorCellIndexRange(domainLo, domainHi);
 
  constCCVariable<double> sigmaT4;
  constCCVariable<double> abskg;
  new_dw->getRegion( abskg   , d_abskgLabel ,   d_matl , level, domainLo, domainHi);
  new_dw->getRegion( sigmaT4 , d_sigmaT4_label, d_matl , level, domainLo, domainHi);
  
  
  double start=clock();

  // patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::rayTrace");

    CCVariable<double> divQ;    
    if( time_sub_step == 0 ){
      new_dw->allocateAndPut( divQ, d_divQLabel, d_matl, patch );
      divQ.initialize( 0.0 );
    }else{
      old_dw->getModifiable( divQ,  d_divQLabel, d_matl, patch );
    }

    double fs;                                         // fraction remaining after all current reflections
    unsigned long int size = 0;                        // current size of PathIndex
    double rho = 1.0 - _alpha;                         // reflectivity
    Vector Dx = patch->dCell();                        // cell spacing


    //__________________________________
    //
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){ 

      IntVector origin = *iter; 
      int i = origin.x();
      int j = origin.y();
      int k = origin.z();

      double chi_Iin_cv = 0;
      double chi = abskg[origin];
      
      // ray loop
      for (int iRay=0; iRay < _NoOfRays; iRay++){

        IntVector cur = origin;

        _mTwister.seed((i + j +k) * iRay +1);        

        double DyDxRatio = Dx.y() / Dx.x(); //noncubic
        double DzDxRatio = Dx.z() / Dx.x(); //noncubic

        Vector ray_location;
        Vector ray_location_prev;
        ray_location[0] =   i +  _mTwister.rand() ;
        ray_location[1] =   j +  _mTwister.rand() * DyDxRatio ; //noncubic
        ray_location[2] =   k +  _mTwister.rand() * DzDxRatio ; //noncubic

        // see http://www.cgafaq.info/wiki/aandom_Points_On_Sphere for explanation

        double plusMinus_one = 2 * _mTwister.rand() - 1;
        double r = sqrt(1 - plusMinus_one * plusMinus_one);    // Radius of circle at z
        double theta = 2 * M_PI * _mTwister.rand();            // Uniform betwen 0-2Pi

        Vector direction_vector;
        direction_vector[0] = r*cos(theta);                   // Convert to cartesian
        direction_vector[1] = r*sin(theta);
        direction_vector[2] = plusMinus_one;                  // Uniform between -1 to 1

        Vector inv_direction_vector = Vector(1.0)/direction_vector;

        int step[3];                                          // Gives +1 or -1 based on sign
        bool sign[3];                
        //bool opposite_sign[3];
        for ( int ii= 0; ii<3; ii++){
          if (inv_direction_vector[ii]>0){
            step[ii] = 1;
            sign[ii] = 1;
          }
          else{
            step[ii] = -1;
            sign[ii] = 0;// 
          }
        }

        double tMaxX = (i + sign[0] - ray_location[0]) * inv_direction_vector[0];
        double tMaxY = (j + sign[1] * DyDxRatio - ray_location[1]) * inv_direction_vector[1]; //noncubic
        double tMaxZ = (k + sign[2] * DzDxRatio - ray_location[2]) * inv_direction_vector[2]; //noncubic

        //Length of t to traverse one cell
        double tDeltaX = abs(inv_direction_vector[0]);
        double tDeltaY = abs(inv_direction_vector[1]) * DyDxRatio; //noncubic
        double tDeltaZ = abs(inv_direction_vector[2]) * DzDxRatio; //noncubic
        double tMax_prev = 0;
        bool in_domain = true;

        //Initializes the following values for each ray
        double intensity = 1.0;     
        double optical_thickness = 0;
        fs = 1;
        
        //+++++++Begin ray tracing+++++++++++++++++++

        Vector temp_direction = direction_vector;   // Used for reflections
        
        //save the direction vector so that it can get modified by...
        //the 2nd switch statement for reflections, but so that we can get the ray_location back into...
        //the domain after it was updated following the first switch statement.         

        //Threshold while loop
        while (intensity > _Threshold){

          //Domain while loop 
          while (in_domain){

            size++;
            IntVector prevCell = cur;
            double disMin = -9;  // Common variable name in ray tracing. Represents ray segment length.

            //__________________________________
            //  Determine which cell the ray will enter next
            if (tMaxX < tMaxY){
              if (tMaxX < tMaxZ){
                cur[0]    = cur[0] + step[0];
                disMin    = tMaxX - tMax_prev;
                tMax_prev = tMaxX;
                tMaxX     = tMaxX + tDeltaX;
              }
              else {
                cur[2]    = cur[2] + step[2];
                disMin    = tMaxZ - tMax_prev;
                tMax_prev = tMaxZ;
                tMaxZ     = tMaxZ + tDeltaZ;
              }
            }
            else {
              if(tMaxY <tMaxZ){
                cur[1]    = cur[1] + step[1];
                disMin    = tMaxY - tMax_prev;
                tMax_prev = tMaxY;
                tMaxY     = tMaxY + tDeltaY;
              }
              else {
                cur[2]    = cur[2] + step[2];
                disMin    = tMaxZ - tMax_prev;
                tMax_prev = tMaxZ;
                tMaxZ     = tMaxZ + tDeltaZ;
              }
            }
            
            in_domain = containsCell(domainLo, domainHi, cur);

            //__________________________________
            //  Update the ray location
            //this is necessary to find the absorb_coef at the endpoints of each step
            ray_location_prev = ray_location;   
            ray_location      = ray_location + (disMin * direction_vector);

            // The running total of alpha*length
            double optical_thickness_prev = optical_thickness;
            optical_thickness += Dx.x() * abskg[prevCell]*disMin; //as long as tDeltaY,Z tMaxY,Z and ray_location[1],[2]..            
                                                                  // were adjusted by DyDxRatio or DzDxRatio, this line is now correct for noncubic domains.  
                              

            intensity = intensity*exp(-optical_thickness);  //update intensity by Beer's Law
            size++;

            //Eqn 3-15(see below reference) while accounting for fs. 
            //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
            chi_Iin_cv += chi * (sigmaT4[prevCell] * ( exp(-optical_thickness_prev) - exp(-optical_thickness) ) * fs );

          } //end domain while loop.  ++++++++++++++

          //__________________________________
          //  Reflections
          if (intensity > _Threshold){

            //puts ray back inside the domain...; 
            intensity*=rho;
            //comment out for cold wall:  Iin_cv += _alpha * Iout_cv[cx] * exp(-optical_thickness)*fs;//!! Right now the temperature of the...
            //boundary is simply the temp of the cell just inside the wall.This is accounting for emission from the walls reacing the origin
            //for non-cold wall, make this a chi_Iin_cv.
            //Comment out for cold, black walls: fs*=rho;//update fs after above Iin reassignment because the reflection is not attenuated by itself.
          }  //end reflection if statement

        }  // end threshold while loop (ends ray tracing for that ray
      }  // Ray loop


      //__________________________________
      //  Compute divQ
      divQ[origin] = 4.0 * _pi * ( sigmaT4[origin] * abskg[origin] - (chi_Iin_cv/_NoOfRays) );
    }  // end cell iterator 

    double end =clock();
    double efficiency = size/((end-start)/ CLOCKS_PER_SEC);
    if (patch->getGridIndex() == 0) {
      cout<< endl;
      cout << " RMCRT REPORT: Patch 0" << endl;
      cout << " Used "<< (end-start) * 1000 / CLOCKS_PER_SEC<< " milliseconds of CPU time. \n" << endl;// Convert time to ms 
      cout << " Size: " << size << endl;
      cout << " Efficiency: " << efficiency << " steps per sec" << endl;
      cout << endl; 
    }
  }  //end patch loop
}  // end ray trace method



//______________________________________________________________________
        inline bool 
Ray::containsCell(const IntVector &low, const IntVector &high, const IntVector &cell)
{
   return  low.x() <= cell.x() && 
           low.y() <= cell.y() &&
           low.z() <= cell.z() &&
           high.x() > cell.x() && 
           high.y() > cell.y() &&
           high.z() > cell.z();
}

//______________________________________________________________________
// ISAAC's NOTES: 
//Jun 9. Ray_noncubic.cc now handles non-cubic cells. Created from Ray.cc as it was in the repository on Jun 9, 2011.
//May 18. cleaned up comments
//May 6. Changed to cell iterator
//Created Jan 31. Cleaned up comments, removed hard coding of T and abskg 
// Jan 19// I changed cx to be lagging.  This changed nothing in the RMS error, but may be important...
//when referencing a non-uniform temperature.
//Created Jan13. //  Ray_PW_const.cc Making this piecewise constant by using CC values. not interpolating
//Removed symmetry test. 
//Has a new equation for absorb_coef for chi and optical thickness calculations...
//I did this based on my findings in my intepolator
//Just commented out a few unnecessary vars
//No more hitch!  Fixed cx to not be incremented the first march, and...
//fixed the formula for absorb_coef and chi which reference ray_location
//Now can do a DelDotqline in each of the three coordinate directions, through the center
//Ray Visualization works, and is correct
//To plot out the rays in matlab
//Now we use an average of two values for a more precise value of absorb_coef rather...
//than using the cell centered absorb_coef
//Using the exact absorb_coef for chi by using formula.Beautiful results...
//see chi_is_exact_absorb_coef.eps in runcases folder
//FIXED THE VARIANCE REDUCTION PROBLEM BY GETTING NEW CHI FOR EACH RAY goes with Chi Fixed folder
//BENCHMARK CASE 99. 
//with error msg if slice is too big
//Based on Ray_bak_Oct15.cc which was Created Oct 13.
// Using Woo (and Amanatides) method//and it works!
//efficiency of approx 20Msteps/sec
//I try to wait to declare each variable until I need it
//Incorporates Steve's sperical way of generating a direction vector
//Back to ijk from cell iterator
//Now absorb_coef is hard coded in because abskg in DW is simply zero
//Now gets abskg from Dw
// with capability to print temperature profile to a file
//Now gets T from DW.  accounts for intensity coming back from surfaces. calculates
// the net Intensity for each cell. Still needs to send rays out from surfaces.Chi is inside while 
//loop. I took out the double domain while loop simply for readability.  I should put it back in 
//when running cases. if(sign[xyorz]) else. See Ray_bak_Aug10.cc for correct implementation. ix 
//is just (NxNyNz) rather than (xNxxNyxNz).  absorbing media. reflections, and simplified while 
//(w/precompute) loop. ray_location now represents what was formally called emiss_point.  It is by
// cell index, not by physical location.
//NOTE equations are from the dissertation of Xiaojing Sun... 
//"REVERSE MONTE CARLO RAY-TRACING FOR RADIATIVE HEAT TRANSFER IN COMBUSTION SYSTEMS 
