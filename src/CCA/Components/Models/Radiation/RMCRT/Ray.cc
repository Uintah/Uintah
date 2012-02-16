//----- Ray.cc ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Components/Models/Radiation/RMCRT/MersenneTwister.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <time.h>

//--------------------------------------------------------------
//
using namespace Uintah;
using namespace std;
static DebugStream dbg("RAY",       false);
static DebugStream dbg2("RAY_DEBUG",false);
static DebugStream dbg_BC("RAY_BC", false);
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

  db->getWithDefault( "NoOfRays"  ,       _NoOfRays  ,      1000 );
  db->getWithDefault( "Threshold" ,       _Threshold ,      0.01 );      // When to terminate a ray
  db->getWithDefault( "Slice"     ,       _slice     ,      9 );         // Level in z direction of xy slice
  db->getWithDefault( "randomSeed",       _isSeedRandom,    true );      // random or deterministic seed.
  db->getWithDefault( "benchmark" ,       _benchmark,       0 );  
  db->getWithDefault("StefanBoltzmann",   _sigma,           5.67051e-8);  // Units are W/(m^2-K)
  db->getWithDefault( "solveBoundaryFlux" , _solveBoundaryFlux, false );
  db->getWithDefault( "CCRays"    ,       _CCRays,          false );  // if true, forces rays to always have CC origins

  if (_benchmark != (0 || 1 || 2 || 3) ){

    ostringstream warn;
    warn << "ERROR:  Benchmark value not set correctly." << endl;
    warn << "Specify a value of 1 through 3 to run a benchmark case, or 0 otherwise." << endl;
    throw InternalError(warn.str(), __FILE__, __LINE__);

  }


  _sigma_over_pi = _sigma/_pi;
  
  const MaterialSubset* mss = d_matlSet->getUnion();
  
  ProblemSpecP root_ps = db->getRootNode();
  
#if 0 
  is_BC_specified(root_ps, d_temperatureLabel->getName(), mss);
  is_BC_specified(root_ps, d_abskgLabel->getName(),       mss);
#endif
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
    CCVariable<double> sigmaT4Pi;

    constCCVariable<double> temperature; 

    if ( time_sub_step == 0 ) { 

      new_dw->allocateAndPut( abskg,    d_abskgLabel,     d_matl, patch ); 
      new_dw->allocateAndPut( sigmaT4Pi,d_sigmaT4_label,  d_matl, patch );
      new_dw->allocateAndPut( absorp,   d_absorpLabel,    d_matl, patch ); 

      abskg.initialize  ( 0.0 ); 
      absorp.initialize ( 0.0 ); 
      sigmaT4Pi.initialize( 0.0 );

      old_dw->get(temperature,      d_temperatureLabel, d_matl, patch, Ghost::None, 0);
    } else { 
      new_dw->getModifiable( sigmaT4Pi, d_sigmaT4_label,  d_matl, patch );
      new_dw->getModifiable( absorp,    d_absorpLabel,    d_matl, patch ); 
      new_dw->getModifiable( abskg,     d_abskgLabel,     d_matl, patch ); 
      new_dw->get( temperature,         d_temperatureLabel, d_matl, patch, Ghost::None, 0 ); 
    }

    IntVector pLow;
    IntVector pHigh;
    level->findInteriorCellIndexRange(pLow, pHigh);

    int Nx = pHigh[0] - pLow[0];
    int Ny = pHigh[1] - pLow[1];
    int Nz = pHigh[2] - pLow[2];

    Vector Dx = patch->dCell(); 


    //__________________________________
    //  Benchmark initializations
    if ( _benchmark == 1 ) {
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( ( c[0] - (Nx - 1.0) /2.0) * Dx[0]) )
                        * ( 1.0 - 2.0 * fabs( ( c[1] - (Ny - 1.0) /2.0) * Dx[1]) )
                        * ( 1.0 - 2.0 * fabs( ( c[2] - (Nz - 1.0) /2.0) * Dx[2]) ) 
                        + 0.1;                  
      }
      // apply boundary conditions
      setBC(abskg, d_abskgLabel->getName(), patch, d_matl);
    }
    else if (_benchmark == 2) {
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){ 
        IntVector c = *iter;
        abskg[c] = 1;
      }
      // apply boundary conditions
      setBC(abskg, d_abskgLabel->getName(), patch, d_matl);
    }    
    else if (_benchmark == 3) {

      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( ( c[0] - (Nx - 1.0) /2.0) * Dx[0]) )
                        * ( 1.0 - 2.0 * fabs( ( c[1] - (Ny - 1.0) /2.0) * Dx[1]) )
                        * ( 1.0 - 2.0 * fabs( ( c[2] - (Nz - 1.0) /2.0) * Dx[2]) ) 
                        + 0.1;  
      }
    }

    //__________________________________
    //  compute sigmaT4

    if(_benchmark == 3) {
      for ( CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ){ 
        IntVector c = *iter; 
        double temp2 = 1000 * abskg[c] * 1000 * abskg[c];
        sigmaT4Pi[c] = _sigma_over_pi * temp2 * temp2; // sigma T^4/pi
      }
    }

    else {
      for ( CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ){ 
        IntVector c = *iter; 
        double temp2 = temperature[c] * temperature[c];
        sigmaT4Pi[c] = _sigma_over_pi * temp2 * temp2; // sigma T^4/pi
      }
    }
  }
}


//---------------------------------------------------------------------------
// 
//---------------------------------------------------------------------------
void
Ray::sched_sigmaT4( const LevelP& level, 
                    SchedulerP& sched,
                    Task::WhichDW temp_dw )
{

  std::string taskname = "Ray::sched_sigmaT4";
  Task* tsk= scinew Task( taskname, this, &Ray::sigmaT4, temp_dw );

  printSchedule(level,dbg,taskname);
  
  tsk->requires( temp_dw, d_temperatureLabel, Ghost::None, 0 ); 
  tsk->computes(d_sigmaT4_label); 

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//---------------------------------------------------------------------------
// Compute total intensity over all wave lengths (sigma * Temperature^4/pi)
//---------------------------------------------------------------------------
void
Ray::sigmaT4( const ProcessorGroup*,
              const PatchSubset* patches,           
              const MaterialSubset*,                
              DataWarehouse* old_dw, 
              DataWarehouse* new_dw,
              Task::WhichDW which_temp_dw )               
{

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::sigmaT4");

    double sigma_over_pi = _sigma/M_PI;

    constCCVariable<double> temp;
    CCVariable<double> sigmaT4;             // sigma T ^4/pi

    DataWarehouse* temp_dw = new_dw->getOtherDataWarehouse(which_temp_dw);
    temp_dw->get(temp,              d_temperatureLabel,   d_matl, patch, Ghost::None, 0);
    new_dw->allocateAndPut(sigmaT4, d_sigmaT4_label,      d_matl, patch);

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
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
Ray::sched_rayTrace( const LevelP& level, 
                     SchedulerP& sched,
                     Task::WhichDW abskg_dw,
                     Task::WhichDW sigma_dw,
                     bool modifies_divQ )
{
  std::string taskname = "Ray::sched_rayTrace";
  Task* tsk= scinew Task( taskname, this, &Ray::rayTrace,
                         modifies_divQ, abskg_dw, sigma_dw );
                         
  printSchedule(level,dbg,taskname);

  // require an infinite number of ghost cells so  you can access
  // the entire domain.
  Ghost::GhostType  gac  = Ghost::AroundCells;
  tsk->requires( abskg_dw , d_abskgLabel  ,  gac, SHRT_MAX);
  tsk->requires( sigma_dw , d_sigmaT4_label, gac, SHRT_MAX);
  //  tsk->requires( Task::OldDW , d_lab->d_cellTypeLabel , Ghost::None , 0 );

  if( modifies_divQ ){
    tsk->modifies( d_divQLabel ); 
  } else {
    tsk->computes( d_divQLabel );
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
               bool modifies_divQ,
               Task::WhichDW which_abskg_dw,
               Task::WhichDW which_sigmaT4_dw )
{ 
  const Level* level = getLevel(patches);
  MTRand _mTwister;


  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  IntVector domainLo_EC, domainHi_EC;
  
  level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
  level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells
  
  DataWarehouse* abskg_dw   = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);

  constCCVariable<double> sigmaT4Pi;
  constCCVariable<double> abskg;                               
  abskg_dw->getRegion(   abskg   ,   d_abskgLabel ,   d_matl , level, domainLo_EC, domainHi_EC);
  sigmaT4_dw->getRegion( sigmaT4Pi , d_sigmaT4_label, d_matl , level, domainLo_EC, domainHi_EC);

  double start=clock();

  // patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::rayTrace");

    CCVariable<double> divQ;    
    if( modifies_divQ ){
      old_dw->getModifiable( divQ,  d_divQLabel, d_matl, patch );
    }else{
      new_dw->allocateAndPut( divQ, d_divQLabel, d_matl, patch );
      divQ.initialize( 0.0 ); 
    }

    unsigned long int size = 0;                        // current size of PathIndex
    Vector Dx = patch->dCell();                        // cell spacing
 

    //__________________________________
    //
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){ 
      IntVector origin = *iter; 
      int i = origin.x();
      int j = origin.y();
      int k = origin.z();

      // Allow for quick debugging test
     /*  IntVector pLow;
       IntVector pHigh;
       level->findInteriorCellIndexRange(pLow, pHigh);
       int Nx = pHigh[0] - pLow[0];
       if (i==Nx/2 && j==Nx/2){
     */  

      double SumI = 0;

      // ray loop
      for (int iRay=0; iRay < _NoOfRays; iRay++){
        IntVector cur = origin;
        IntVector prevCell = cur;

        if(_isSeedRandom == false){
          _mTwister.seed((i + j +k) * iRay +1);
        }

        // see http://www.cgafaq.info/wiki/aandom_Points_On_Sphere for explanation

        double plusMinus_one = 2 * _mTwister.rand() - 1;
        double r = sqrt(1 - plusMinus_one * plusMinus_one);    // Radius of circle at z
        double theta = 2 * M_PI * _mTwister.rand();            // Uniform betwen 0-2Pi

        Vector direction_vector;
        direction_vector[0] = r*cos(theta);                   // Convert to cartesian
        direction_vector[1] = r*sin(theta);
        direction_vector[2] = plusMinus_one;                  
        Vector inv_direction_vector = Vector(1.0)/direction_vector;

        int step[3];                                          // Gives +1 or -1 based on sign
        bool sign[3];
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
        
        double DyDxRatio = Dx.y() / Dx.x(); //noncubic
        double DzDxRatio = Dx.z() / Dx.x(); //noncubic

        Vector ray_location;
        Vector ray_location_prev;

        if(_CCRays){
          ray_location[0] =   i +  0.5 ;
          ray_location[1] =   j +  0.5 * DyDxRatio ; //noncubic
          ray_location[2] =   k +  0.5 * DzDxRatio ; //noncubic

        }

        else{
          ray_location[0] =   i +  _mTwister.rand() ;
          ray_location[1] =   j +  _mTwister.rand() * DyDxRatio ; //noncubic
          ray_location[2] =   k +  _mTwister.rand() * DzDxRatio ; //noncubic
        }


        double tMaxX = (i + sign[0]             - ray_location[0]) * inv_direction_vector[0];
        double tMaxY = (j + sign[1] * DyDxRatio - ray_location[1]) * inv_direction_vector[1];
        double tMaxZ = (k + sign[2] * DzDxRatio - ray_location[2]) * inv_direction_vector[2];

        //Length of t to traverse one cell
        double tDeltaX = abs(inv_direction_vector[0]);
        double tDeltaY = abs(inv_direction_vector[1]) * DyDxRatio;
        double tDeltaZ = abs(inv_direction_vector[2]) * DzDxRatio;
        double tMax_prev = 0;
        bool in_domain = true;

        //Initializes the following values for each ray
        double intensity = 1.0;
        double fs = 1.0;
        double optical_thickness = 0;

        //+++++++Begin ray tracing+++++++++++++++++++

        Vector temp_direction = direction_vector;   // Used for reflections

        //save the direction vector so that it can get modified by...
        //the 2nd switch statement for reflections, but so that we can get the ray_location back into...
        //the domain after it was updated following the first switch statement.

        int nReflect = 0; // Number of reflections that a ray has undergone
        //Threshold while loop
        while (intensity > _Threshold){
        
          int face = -9;

          while (in_domain){
            
            prevCell = cur;
            double disMin = -9;  // Common variable name in ray tracing. Represents ray segment length.

            //__________________________________
            //  Determine which cell the ray will enter next
            if (tMaxX < tMaxY){
              if (tMaxX < tMaxZ){
                cur[0]    = cur[0] + step[0];
                disMin    = tMaxX - tMax_prev;
                tMax_prev = tMaxX;
                tMaxX     = tMaxX + tDeltaX;
                face      = 0;
              }
              else {
                cur[2]    = cur[2] + step[2];
                disMin    = tMaxZ - tMax_prev;
                tMax_prev = tMaxZ;
                tMaxZ     = tMaxZ + tDeltaZ;
                face      = 2;
              }
            }
            else {
              if(tMaxY <tMaxZ){
                cur[1]    = cur[1] + step[1];
                disMin    = tMaxY - tMax_prev;
                tMax_prev = tMaxY;
                tMaxY     = tMaxY + tDeltaY;
                face      = 1;
              }
              else {
                cur[2]    = cur[2] + step[2];
                disMin    = tMaxZ - tMax_prev;
                tMax_prev = tMaxZ;
                tMaxZ     = tMaxZ + tDeltaZ;
                face      =2;
              }
            }

            in_domain = containsCell(domainLo, domainHi, cur, face);

            //__________________________________
            //  Update the ray location
            //this is necessary to find the absorb_coef at the endpoints of each step if doing interpolations
            //ray_location_prev = ray_location;
            //ray_location      = ray_location + (disMin * direction_vector);// If this line is used,  make sure that direction_vector is adjusted after a reflection

            // The running total of alpha*length
            double optical_thickness_prev = optical_thickness;
            optical_thickness += Dx.x() * abskg[prevCell]*disMin; //as long as tDeltaY,Z tMaxY,Z and ray_location[1],[2]..
            // were adjusted by DyDxRatio or DzDxRatio, this line is now correct for noncubic domains.


            size++;

            //Eqn 3-15(see below reference) while
            //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
            SumI += sigmaT4Pi[prevCell] * ( exp(-optical_thickness_prev) - exp(-optical_thickness) ) * fs;
          } //end domain while loop.  ++++++++++++++

          intensity = exp(-optical_thickness);

          //  wall emission 12/15/11
          SumI += abskg[cur]*sigmaT4Pi[cur] * intensity;

          intensity = intensity * (1-abskg[cur]);

          //__________________________________
          //  Reflections
          if (intensity > _Threshold){

            ++nReflect;
            fs = fs * (1-abskg[cur]);

            //put cur back inside the domain
            cur = prevCell;

            // apply reflection condition
            step[face] *= -1;                      // begin stepping in opposite direction
            sign[face] = (sign[face]==1) ? 0 : 1; //  swap sign from 1 to 0 or vice versa

            in_domain = 1;


          }  // if reflection
        }  // threshold while loop.
      }  // Ray loop

      //__________________________________
      //  Compute divQ
      divQ[origin] = 4.0 * _pi * abskg[origin] * ( sigmaT4Pi[origin] - (SumI/_NoOfRays) );
      //cout << divQ[origin] << endl;
     // } // end quick debug testing
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



//---------------------------------------------------------------------------
// Ray tracing using the multilevel data onion scheme
//---------------------------------------------------------------------------
void
Ray::sched_rayTrace_dataOnion( const LevelP& level, 
                               SchedulerP& sched,
                               Task::WhichDW abskg_dw,
                               Task::WhichDW sigma_dw,
                               bool modifies_divQ )
{
  int maxLevels = level->getGrid()->numLevels() -1;
  int L_indx = level->getIndex();
  
  if(L_indx != maxLevels){     // only schedule on the finest level
    return;
  }
  std::string taskname = "Ray::sched_rayTrace_dataOnion";
  Task* tsk= scinew Task( taskname, this, &Ray::rayTrace_dataOnion,
                          modifies_divQ, abskg_dw, sigma_dw );
                          
  printSchedule(level,dbg,taskname);

  Task::DomainSpec  ND  = Task::NormalDomain;
  #define allPatches 0
  #define allMatls 0
  Ghost::GhostType  gn  = Ghost::None;
  Ghost::GhostType  gac  = Ghost::AroundCells;
  bool  fat = true;  // possibly (F)rom (A)nother (T)askgraph
  
  // finest level
  tsk->requires(abskg_dw, d_abskgLabel,     gn, 0);
  tsk->requires(sigma_dw, d_sigmaT4_label,  gn, 0);
  
  // coarser levels
  tsk->requires(abskg_dw, d_abskgLabel,     allPatches, Task::CoarseLevel,allMatls, ND, gac, SHRT_MAX);
  tsk->requires(sigma_dw, d_sigmaT4_label,  allPatches, Task::CoarseLevel,allMatls, ND, gac, SHRT_MAX);
  
  if( modifies_divQ ){
    tsk->modifies( d_divQLabel );
  } else {
    
    tsk->computes( d_divQLabel );
  }
  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}


//---------------------------------------------------------------------------
// Ray tracer using the multilevel "data onion" scheme
//---------------------------------------------------------------------------
void
Ray::rayTrace_dataOnion( const ProcessorGroup* pc,
                         const PatchSubset* finePatches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         bool modifies_divQ,
                         Task::WhichDW which_abskg_dw,
                         Task::WhichDW which_sigmaT4_dw )
{ 
  const Level* fineLevel = getLevel(finePatches);
  int maxLevels    = fineLevel->getGrid()->numLevels();
  int levelPatchID = fineLevel->getPatch(0)->getID();
  MTRand _mTwister;

  //__________________________________
  //retrieve all of the data for all levels
  StaticArray< constCCVariable<double> > abskg(maxLevels);
  StaticArray< constCCVariable<double> >sigmaT4Pi(maxLevels);
 
  DataWarehouse* abskg_dw   = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);
  
  vector<Vector> Dx(maxLevels);
  double DyDx[maxLevels];
  double DzDx[maxLevels];
  
  for(int L = 0; L<maxLevels; L++){
    LevelP level = new_dw->getGrid()->getLevel(L);
    
    if( level->hasFinerLevel() ){
      IntVector domainLo_EC, domainHi_EC;
      level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells

      abskg_dw->getRegion(   abskg[L]   ,   d_abskgLabel ,   d_matl , level.get_rep(), domainLo_EC, domainHi_EC);
      sigmaT4_dw->getRegion( sigmaT4Pi[L] , d_sigmaT4_label, d_matl , level.get_rep(), domainLo_EC, domainHi_EC);
    }
    
    Vector dx = level->dCell();
    DyDx[L] = dx.y() / dx.x();
    DzDx[L] = dx.z() / dx.x();
    Dx[L] = dx;
  }
  
  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  fineLevel->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells

  double start=clock();

  //__________________________________
  //patch loop
  for (int p=0; p < finePatches->size(); p++){

    const Patch* finePatch = finePatches->get(p);
    printTask(finePatches, finePatch,dbg,"Doing Ray::rayTrace_dataOnion");

    CCVariable<double> divQ_fine;  
    constCCVariable<double> abskg_fine;
    constCCVariable<double> sigmaT4Pi_fine;
    Ghost::GhostType  gn  = Ghost::None;
    
    if( modifies_divQ ){
      old_dw->getModifiable( divQ_fine,  d_divQLabel, d_matl, finePatch );
    }else{
      new_dw->allocateAndPut( divQ_fine, d_divQLabel, d_matl, finePatch );
      divQ_fine.initialize( 0.0 );
    }

    abskg_dw->get(   abskg_fine,      d_abskgLabel,   d_matl, finePatch, gn, 0);
    sigmaT4_dw->get( sigmaT4Pi_fine,  d_sigmaT4_label,d_matl, finePatch, gn, 0);
    
    abskg[maxLevels-1]     = abskg_fine;
    sigmaT4Pi[maxLevels-1] = sigmaT4Pi_fine;
    
    

    unsigned long int size = 0;                             // current size of PathIndex

    //__________________________________
    //
    for (CellIterator iter = finePatch->getCellIterator(); !iter.done(); iter++){ 

      IntVector origin = *iter; 
      int i = origin.x();
      int j = origin.y();
      int k = origin.z();
      
      // Allow for quick debugging test
     /*  IntVector pLow;
       IntVector pHigh;
       level->findInteriorCellIndexRange(pLow, pHigh);
       int Nx = pHigh[0] - pLow[0];
       if (i==Nx/2 && k==Nx/2){
     */

      double SumI = 0;
      
      Vector tMax;
      vector<Vector> tDelta(maxLevels);

      //__________________________________
      //  ray loop
      for (int iRay=0; iRay < _NoOfRays; iRay++){
        IntVector cur      = origin;
        IntVector prevCell = cur;
        
        int L       = maxLevels -1;  // finest level
        int prevLev = L;
        

        if(_isSeedRandom == false){
          _mTwister.seed((i + j +k) * iRay +1);
        }

        //__________________________________
        //  Ray direction      
        // see http://www.cgafaq.info/wiki/aandom_Points_On_Sphere for explanation

        double plusMinus_one = 2 * _mTwister.rand() - 1;
        double r = sqrt(1 - plusMinus_one * plusMinus_one);    // Radius of circle at z
        double theta = 2 * M_PI * _mTwister.rand();            // Uniform betwen 0-2Pi

        // dbg2 << " plusMinus_one " << plusMinus_one << " r " << r << " theta " << theta << endl;

        Vector direction;
        direction[0] = r*cos(theta);                           // Convert to cartesian
        direction[1] = r*sin(theta);
        direction[2] = plusMinus_one;                  
        Vector inv_direction = Vector(1.0)/direction;

        int step[3];                                           // Gives +1 or -1 based on sign
        bool sign[3];
        for ( int ii= 0; ii<3; ii++){
          if (inv_direction[ii]>0){
            step[ii] = 1;
            sign[ii] = 1;
          }
          else{
            step[ii] = -1;
            sign[ii] = 0;
          }
        }
        
        //__________________________________
        // define tMax & tDelta on all levels
        // go from finest to coarset level so you can compare 
        // with 1L rayTrace results.
       // Vector randNum = Vector(_mTwister.rand(), _mTwister.rand(), _mTwister.rand());
        
        tMax.x( (sign[0]  - _mTwister.rand())            * inv_direction[0] );  
        tMax.y( (sign[1]  - _mTwister.rand()) * DyDx[L]  * inv_direction[1] );  
        tMax.z( (sign[2]  - _mTwister.rand()) * DzDx[L]  * inv_direction[2] );  
        
        for(int Lev = maxLevels-1; Lev>-1; Lev--){
          //Length of t to traverse one cell
          tDelta[Lev].x( abs(inv_direction[0]) );
          tDelta[Lev].y( abs(inv_direction[1]) * DyDx[Lev] );
          tDelta[Lev].z( abs(inv_direction[2]) * DzDx[Lev] );
        }

        //Initializes the following values for each ray
        bool   in_domain      = true;
        double tMax_prev      = 0;
        double intensity      = 1.0;
        double fs             = 1.0;
        int    nReflect       = 0;             // Number of reflections
        double optical_thickness = 0;
        bool   onFinePatch    = true;
        const Level* level    = fineLevel;

        //______________________________________________________________________
        //  Threshold  loop
        while (intensity > _Threshold){
          
          int face = -9;
          
          while (in_domain){
            
            prevCell = cur;
            prevLev  = L;
            
            double disMin = -9;   // Ray segment length.
            
            //__________________________________
            //  Determine the princple direction the ray is traveling
            //
            DIR dir = NONE;
            if (tMax.x() < tMax.y()){
              if (tMax.x() < tMax.z()){
                dir = X;
              } else {
                dir = Z;
              }
            }
            else {
              if(tMax.y() <tMax.z()){
                dir = Y;
              } else {
                dir = Z;
              }
            }
            
            // next cell
            cur[dir]  = cur[dir] + step[dir];
            
            //__________________________________
            // Logic for moving between levels
            // currently you can only move  from fine to coarse level
            if( onFinePatch && finePatch->containsCell(cur) == false ){
              cur   = level->mapCellToCoarser(cur); 
              level = level->getCoarserLevel().get_rep();
              level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
              L     = level->getIndex();
              onFinePatch = false;
              dbg2 << " Jumping off fine patch switching Levels:  prev L: " << prevLev << " cur L " << L << " cur " << cur << endl;
              
            }    // TODD: CLEAN THIS UP
            else if ( onFinePatch == false && level->containsCell(cur) == false && L > 0 ){
              cout << " Trying to switch : L " << L << endl;
              cur   = level->mapCellToCoarser(cur); 
              level = level->getCoarserLevel().get_rep();
              level->findInteriorCellIndexRange(domainLo, domainHi); 
              L     = level->getIndex();
              dbg2 << " Switching Levels:  prev L: " << prevLev << " cur L " << L << " cur " << cur << endl;
            }
            
            //__________________________________
            //  update marching variables
            disMin        = tMax[dir] - tMax_prev;        // Todd:   replace tMax[dir]
            tMax_prev     = tMax[dir];
            tMax[dir]     = tMax[dir] + tDelta[L][dir];
            face          = dir;
     

            in_domain = containsCell(domainLo, domainHi, cur, face);

            //__________________________________
            //  Update the ray location
            //this is necessary to find the absorb_coef at the endpoints of each step if doing interpolations
            //ray_location_prev = ray_location;
            //ray_location      = ray_location + (disMin * direction_vector);// If this line is used,  make sure that direction_vector is adjusted after a reflection

            // The running total of alpha*length
            double optical_thickness_prev = optical_thickness;
            optical_thickness += Dx[prevLev].x() * abskg[prevLev][prevCell]*disMin; //as long as tDelta[L].y(),Z tMax.y(),Z and ray_location[1],[2]..
            // were adjusted by DyDxRatio or DzDxRatio, this line is now correct for noncubic domains.

            size++;

            //Eqn 3-15(see below reference) while
            //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
            SumI += sigmaT4Pi[prevLev][prevCell] * ( exp(-optical_thickness_prev) - exp(-optical_thickness) ) * fs;
            
         //   dbg2 << "origin " << origin << "dir " << dir << " cur " << cur <<" prevCell " << prevCell << " sumI " << SumI << " in_domain " << in_domain << endl;
         //   dbg2 << "    tmaxX " << tMax[L].x() << " tmaxY " << tMax[L].y() << " tmaxZ " << tMax[L].z() << endl;
         //   dbg2 << "    direction " << direction << endl;
         
          } //end domain while loop.  ++++++++++++++

          intensity = exp(-optical_thickness);

          //  wall emission 12/15/11
          SumI += abskg[L][cur] * sigmaT4Pi[L][cur] * intensity;

          intensity = intensity * (1-abskg[L][cur]);

          //__________________________________
          //  Reflections
          if (intensity > _Threshold){

            ++nReflect;
            fs = fs * (1 - abskg[L][cur]);

            //put cur back inside the domain
            cur = prevCell;
            in_domain = 1;

            // apply reflection condition
            step[face] *= -1;                      // begin stepping in opposite direction
            sign[face] = (sign[face]==1) ? 0 : 1;  //  swap sign from 1 to 0 or vice versa
            dbg2 << " REFLECTING " << endl;
          }  // if reflection
        }  // threshold while loop.
      }  // Ray loop

      //__________________________________
      //  Compute divQ
      divQ_fine[origin] = 4.0 * _pi * abskg_fine[origin] * ( sigmaT4Pi_fine[origin] - (SumI/_NoOfRays) );
      dbg2 << origin << "    divQ: " << divQ_fine[origin] << " term2 " << abskg_fine[origin] << " sumI term " << (SumI/_NoOfRays) << endl;
      
      //cout << divQ[origin] << endl;
       // } // end quick debug testing
    }  // end cell iterator

    double end =clock();
    double efficiency = size/((end-start)/ CLOCKS_PER_SEC);
    if (finePatch->getGridIndex() == levelPatchID) {
      cout<< endl;
      cout << " RMCRT REPORT: Patch " << levelPatchID <<endl;
      cout << " Used "<< (end-start) * 1000 / CLOCKS_PER_SEC<< " milliseconds of CPU time. \n" << endl;// Convert time to ms
      cout << " Size: " << size << endl;
      cout << " Efficiency: " << efficiency << " steps per sec" << endl;
      cout << endl;
    }
  }  //end finePatch loop
}  // end ray trace method


//______________________________________________________________________
inline bool
Ray::containsCell(const IntVector &low, const IntVector &high, const IntVector &cell, const int &face)
{
  return  low[face] <= cell[face] &&
          high[face] > cell[face];
}


//---------------------------------------------------------------------------
// 
//---------------------------------------------------------------------------
void
Ray::sched_setBoundaryConditions( const LevelP& level, 
                                  SchedulerP& sched )
{

  std::string taskname = "Ray::sched_setBoundaryConditions";
  Task* tsk= scinew Task( taskname, this, &Ray::setBoundaryConditions );

  printSchedule(level,dbg,taskname);

  tsk->modifies(d_sigmaT4_label); 
  tsk->modifies(d_abskgLabel);

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//---------------------------------------------------------------------------
void
Ray::setBoundaryConditions( const ProcessorGroup*,
                            const PatchSubset* patches,           
                            const MaterialSubset*,                
                            DataWarehouse*,                
                            DataWarehouse* new_dw )               
{

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    
    if( bf.size() > 0){
    
      printTask(patches,patch,dbg,"Doing Ray::setBoundaryConditions");

      double sigma_over_pi = _sigma/M_PI;

      CCVariable<double> temp;
      CCVariable<double> abskg;
      CCVariable<double> sigmaT4Pi;

      new_dw->allocateTemporary(temp,  patch);
      new_dw->getModifiable( abskg,     d_abskgLabel,    d_matl, patch );
      new_dw->getModifiable( sigmaT4Pi, d_sigmaT4_label,  d_matl, patch );

      setBC(abskg, d_abskgLabel->getName(),       patch, d_matl);
      setBC(temp,  d_temperatureLabel->getName(), patch, d_matl);

      //__________________________________
      // loop over boundary faces and compute sigma T^4
      for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
        Patch::FaceType face = *itr;

        Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

        for(CellIterator iter=patch->getFaceIterator(face, PEC); !iter.done();iter++) {
          const IntVector& c = *iter;
          double T_sqrd = temp[c] * temp[c];
          sigmaT4Pi[c] = sigma_over_pi * T_sqrd * T_sqrd;
        }
      } 
    } // has a boundaryFace
  }
}

//______________________________________________________________________
//  Set Boundary conditions
void 
Ray::setBC(CCVariable<double>& Q_CC,
       const string& desc,
       const Patch* patch,
       const int mat_id)
{
  if(patch->hasBoundaryFaces() == false){
    return;
  }

  dbg_BC << "setBC \t"<< desc <<" "
        << " mat_id = " << mat_id <<  ", Patch: "<< patch->getID() << endl;

  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    int nCells = 0;
    string bc_kind = "NotSet";

    IntVector dir= patch->getFaceAxes(face);
    Vector cell_dx = patch->dCell();
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    // iterate over each geometry object along that face
    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;

      bool foundIterator = 
        getIteratorBCValueBCKind( patch, face, child, desc, mat_id,
                        bc_value, bound_ptr,bc_kind); 

      if(foundIterator) {

        //__________________________________
        // Dirichlet
        if(bc_kind == "Dirichlet"){
          nCells += setDirichletBC_CC<double>( Q_CC, bound_ptr, bc_value);
        }
        //__________________________________
        // Neumann
        else if(bc_kind == "Neumann"){
          nCells += setNeumannBC_CC<double>( patch, face, Q_CC, bound_ptr, bc_value, cell_dx);
        }                                   
        //__________________________________
        //  Symmetry
        else if ( bc_kind == "symmetry" || bc_kind == "zeroNeumann" ) {
          bc_value = 0.0;
          nCells += setNeumannBC_CC<double>( patch, face, Q_CC, bound_ptr, bc_value, cell_dx);
        }

        //__________________________________
        //  debugging
        if( dbg_BC.active() ) {
          bound_ptr.reset();
          dbg_BC <<"Face: "<< patch->getFaceName(face) <<" numCellsTouched " << nCells
             <<"\t child " << child  <<" NumChildren "<<numChildren 
             <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
             <<"\t bound limits = "<< bound_ptr << endl;
        }
      }  // if iterator found
    }  // child loop

    dbg_BC << "    "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
               << " nCellsTouched: " << nCells << endl;
    //__________________________________
    //  bulletproofing
#if 0
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);

    if(nCells != nFaceCells){
      ostringstream warn;
      warn << "ERROR: ICE: setSpecificVolBC Boundary conditions were not set correctly ("<< desc<< ", " 
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells<<") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
#endif
  }  // faces loop
}


//______________________________________________________________________
// ISAAC's NOTES: 
//Jan 6. Began work on solving for boundary fluxes
//Jan 5. Changed containsCell method to only need to compare two faces rather than 6
//Dec 15. Now uses interactive BCs correctly from input file
//Dec 1. Clean up (removed ray viz stuff.
//Nov 30. Modified so user can specify in the input file either benchmark_13pt2, benchmark_1, or no benchmark (real case)
//Nov 18. Put in visualization stuff. It worked well.
//Nov 16. Realized that the "correct" method for reflections is no different from Paula's (using fs). Reverted back to Paula's
//Nov 9, 2011.  Added in reflections based on correct method of attenuating I for each reflection.
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
