//----- Ray.cc ----------------------------------------------
#include "CCA/Components/Arches/RMCRT/Ray.h"
#include <time.h>
//--------------------------------------------------------------
//
using namespace Uintah;
using namespace std;

//---------------------------------------------------------------------------
// Method: Constructor. he's not creating an instance to the class yet
//---------------------------------------------------------------------------
Ray::Ray( const ArchesLabel* labels ):
  d_lab(labels)
{
  _pi = acos(-1); 
}

//---------------------------------------------------------------------------
// Method: Destructor
//---------------------------------------------------------------------------
Ray::~Ray()
{
}

//---------------------------------------------------------------------------
// Method: Problem setup (access to input file information)
//---------------------------------------------------------------------------
void
Ray::problemSetup( const ProblemSpecP& inputdb ) 
{
  ProblemSpecP db = inputdb;

  db->getWithDefault( "NoOfRays", d_NoOfRays, 1000 );
  db->getWithDefault( "Threshold", d_Threshold, 0.01 );  //When to terminate a ray
  db->getWithDefault( "Alpha", _alpha, 0.2 );            //Absorption coefficient of the boundaries
  db->getWithDefault( "Slice", _slice, 9 );              //Level in z direction of xy slice
}

//---------------------------------------------------------------------------
// Method: Schedule the ray tracer
//---------------------------------------------------------------------------
  void
Ray::sched_rayTrace( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "Ray::rayTrace";
  Task* tsk= scinew Task( taskname, this, &Ray::rayTrace );

  tsk->requires( Task::OldDW, d_lab->d_tempINLabel,   Ghost::None, 0 ); 
  tsk->requires( Task::OldDW, d_lab->d_abskgINLabel,  Ghost::None, 0 );
  tsk->requires( Task::OldDW, d_lab->d_cellTypeLabel, Ghost::None, 0 );
  
  tsk->computes(d_lab->d_RMCRT_fixMeLabel); 

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() );
}

//---------------------------------------------------------------------------
// Method: The actual work of the ray tracer
//---------------------------------------------------------------------------
void
Ray::rayTrace( const ProcessorGroup* pc,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw )
{

  double start=clock();

  // patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    //int index = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    //The following will be removed when using cell iterator..
    int cur[3];//  cur represents the current location of a ray as it is traced through the domain... 
    //Each ray will always begin at c (the i,j,k of the iterator)
    Vector ray_location;
    Vector ray_location_prev;

    // since we are only requiring temperature, we get it as a const.  !!May need to remove the const part...
    //because *ix_ptr will be changed.
    constCCVariable<double> temperature;
    constCCVariable<double> abskg;
    CCVariable<double> fixMe;

    double disMin;

    // CCVariable<double> *array_ptr = temperature; // this is the pointer to the start of the 3D "CCVariable" array
    // CCVariable<double> *ix_ptr = array_ptr; // this pointer is free to move along the members of the array

    //  getting the temperature from the DW
    old_dw->get(temperature,      d_lab->d_tempINLabel,       matlIndex, patch, Ghost::None, 0);
    old_dw->get(abskg,            d_lab->d_abskgINLabel,      matlIndex, patch, Ghost::None, 0);
    new_dw->allocateAndPut(fixMe, d_lab->d_RMCRT_fixMeLabel,  matlIndex, patch);
    //old_dw->getModifiable(IsaacFlux, d_lab->d_radiationVolqINIsaacLabel, index, patch);

    fixMe.initialize(0.0);
  
    IntVector pLow  = patch->getCellLowIndex();  // patch low index //returns 0 for edge patches
    IntVector pHigh = patch->getCellHighIndex(); // patch high index//returns index of highest cell (usually it's a ghost)
    int ii;//used as an index when creating 1D arrays out of 3d arrays

    int Nx = pHigh[0] - pLow[0];
    int Ny = pHigh[1] - pLow[1];
    int Nz = pHigh[2] - pLow[2];   
    int ix =(Nx)*(Ny)*(Nz);

    //For visualization of rays in a single line in Nx X Ny slice
    /*
    double RayVisX[d_NoOfRays*2][Nx*3];
    double RayVisY[d_NoOfRays*2][Nx*3];
    double RayVisZ[d_NoOfRays*2][Nx*3];


    //For visualization of Rays only. Initialize

    for (int rRay=0; rRay < d_NoOfRays*2; rRay++){
      for (int qx=0; qx < Nx*3; qx++){
        RayVisX[rRay][qx] = -1.0;
        RayVisY[rRay][qx] = -1.0;
        RayVisZ[rRay][qx] = -1.0;   
      }  
    }
    */

    double absorb_coef[ix]; //represents the fraction absorbed per cell length through a control volume 
    //the data warehouse values begin at zero
    double dx_absorb_coef[ix];// Dx multiplied by the absorption coefficient
    double Iout_cv[ix];//Array because each cell's intensity will be referenced so many...
    //times through all the paths of other cell's rays
    double chi_Iin_cv;//  Iin multiplied by its respective chi. 
    double Inet_cv[ix];// separate Inet necessary for surfaces. !! I don't think I need an array for this
    //double abskg1D[ix];// !! used to visualize abskg only.  otherwise comment out
    double sigma_over_pi = 1.804944378616567e-8;//Stefan Boltzmann divided by pi (W* m-2* K-4)
    double chi; //the absorption coefficient multiplied of the origin cell 
    bool   first;//used to determine chi. 
    double fs; // fraction remaining after all current reflections

    unsigned int size = 0;//current size of PathIndex !!move to Ray.h
    //int netFaceFlux[ix][6]; // net radiative flux on each face of each cell //0:_bottom, 1:_top; 2:_south, 3:_north; 4:_west, 5:_east
    double rho = 1.0 - _alpha; //reflectivity
    double optical_thickness;//The running total of alpha*length !!move to Ray.h  
    double optical_thickness_prev;//The running total of alpha*length !!move to Ray.h     
    Vector Dx = patch->dCell(); // cell spacing
    IntVector   c; //represents i, j, k
    const double* temperature_ptr = const_cast<double*>( temperature.getPointer() );//maybe tempINLabel??
    const double* abskg_ptr = const_cast<double*>( abskg.getPointer() );//maybe tempINLabel??
    //double absorb_coef_3D[Nz][Ny][Nx];

    //Make Iout a 1D while referencing temperature which is a 3D array. This pre-computes Iout since it gets referenced so much
    //Pointer way
    ii=0;    
    for ( k=0; k<Nz; k++){//the indeces of the cells
      for ( j=0; j<Ny; j++){
        for ( i=0; i<Nx; i++){
          Iout_cv[ii] = (*temperature_ptr) * (*temperature_ptr) * (*temperature_ptr) * (*temperature_ptr) * sigma_over_pi;//sigmaT^4/pi
          absorb_coef[ii] = *abskg_ptr;//make 1D from 3D
          //for benchmark case, temperature is 64.804 everywhere
          //Iout_cv[ii] = 64.804361 * 64.804361 * 64.804361 * 64.804361 * sigma_over_pi;//T^4*sigma/pi
         // absorb_coef[ii] = 0.9*(1-2*fabs((i -(Nx-1)/2.0)*Dx[0]))*(1-2*fabs((j -(Ny-1)/2.0)*Dx[1]))*(1-2*fabs((k -(Nz-1)/2.0)*Dx[2])) +0.1;//benchmark 99
          //next line is for visualization only.
          dx_absorb_coef[ii] = Dx.x()*absorb_coef[ii];//Used in optical thickness calculation.!!Adjust if cells are noncubic
          temperature_ptr++;
          abskg_ptr++;
          ii++;
        }
      }
    }

    ix = 0;  //This now represents the current cell in 1D(akin to c, not cur)

    // int i_flag = 1;//visualization of chi_Iin only
    // int rRay = 0;//visualization of rays.
    // cell loop
    for ( k=0; k<Nz; k++){
      for ( j=0; j<Ny; j++){
        for ( i=0; i<Nx; i++){
          chi_Iin_cv = 0; 

          // ray loop
          for (int iRay=0; iRay < d_NoOfRays; iRay++){ //counter. goes from 0 to NoOfRays
            //if (k==_slice && j==_slice && i>18)  
            //        int qx = 0; //for visualization

            cur[0] = i; cur[1] =j; cur[2] = k; //Use cur = c when cur is an IntVector, when using cell iterato
            //IntVector cur = c;//current will represent the current location of a ray as it is traced through the...
            //domain.  Each ray will always begin at c (the i,j,k of the iterator)
            int cx = ix;//
            int cx_p;
            // begins at 0. Can follow a ray without touching ix.(akin to cur)
            // picks a random spot in the cell, starting from the negative face:
            _mTwister.seed((i + j + k) * iRay +1);

#if 0           
            double a = _mTwister.rand();
            double b = _mTwister.rand();
            double c = _mTwister.rand();
            cout << IntVector(i,j,k) << "randomNumbers " << a << " " << b << " " << c << endl;
#endif            
            ray_location[0] =   i +  _mTwister.rand() ;//was c.x +...
            ray_location[1] =   j +  _mTwister.rand() ;
            ray_location[2] =   k +  _mTwister.rand() ; //Dx is specified in line 101

            

            /*      //this is for visualization of the rays only.  Otherwise comment out.

                    if (k==_slice && j==_slice && (i==19 || i ==21) ){
                    RayVisX[rRay][qx]=ray_location[0];
                    RayVisY[rRay][qx]=ray_location[1];
                    RayVisZ[rRay][qx]=ray_location[2];
                    if (i==21) cout << endl;
                    cout << endl << ray_location<<  " cx " << cx << endl;
                    qx++;
                    }

             */
            //ray_location = + _mTwister.randVector(); Sep 27, get this to work.  I need cur to be a vector
            // see http://www.cgafaq.info/wiki/aandom_Points_On_Sphere for explanation
            Vector direction_vector;//change to capital Vector
            direction_vector[2] = 2 * _mTwister.rand() - 1;  // Uniform between -1 to 1
            double r = sqrt(1 - direction_vector[2]*direction_vector[2]); // Radius of circle at z
            double theta = 2*_pi*_mTwister.rand(); // Uniform betwen 0-2Pi
            direction_vector[0] = r*cos(theta); // Convert to cartesian
            direction_vector[1] = r*sin(theta);
            Vector inv_direction_vector;


            inv_direction_vector[0] = 1.0/direction_vector[0];
            inv_direction_vector[1] = 1.0/direction_vector[1];
            inv_direction_vector[2] = 1.0/direction_vector[2];
            //inv_direction_vector = Vector(1)/direction_vector;  //!! try this way, easier to read

            int step[3];//this can be a Vector in arches. Gives +1 or -1 based on sign
            bool sign[3];
            //bool opposite_sign[3];
            for ( ii= 0; ii<3; ii++){
              if (inv_direction_vector[ii]>0){
                step[ii] = 1;
                sign[ii] = 1;
                //  opposite_sign[ii]=0;
              }
              else{
                step[ii] = -1;
                sign[ii] = 0;// 
                //opposite_sign[ii]=1;
              }
            }

            double tMaxX = (i +sign[0] - ray_location[0]) * inv_direction_vector[0];
            double tMaxY = (j +sign[1] - ray_location[1]) * inv_direction_vector[1];
            double tMaxZ = (k +sign[2] - ray_location[2]) * inv_direction_vector[2];


            double tDeltaX = abs(inv_direction_vector[0]);//Tells us the lenght of t to traverse one cell
            double tDeltaY = abs(inv_direction_vector[1]);
            double tDeltaZ = abs(inv_direction_vector[2]);
            double tMax_prev = 0;
            bool in_domain = 1;

            //Initializes the following values for each ray
            double intensity = 1.0;     
            optical_thickness = 0;
            first = 1;
            fs = 1;
            //+++++++Begin ray tracing+++++++++++++++++++

            Vector temp_direction = direction_vector;//save the direction vector so that it can get modified by...
            //the 2nd switch statement for reflections, but so that we can get the ray_location back into...
            //the domain after it was updated following the first switch statement.         

            //Threshold while loop
            while (intensity > d_Threshold){

              //Domain while loop 
              while (in_domain){
                size++;
                cx_p =cx;//cx previous.  Jan 19
                if (tMaxX < tMaxY){
                  if (tMaxX < tMaxZ){
                    cx = cx +step[0];
                    cur[0] += step[0];
                    if (cur[0] > Nx-1 || cur[0] < 0){
                      in_domain = 0; 
                    }
                    disMin = tMaxX - tMax_prev;
                    tMax_prev = tMaxX;
                    tMaxX = tMaxX +tDeltaX;
                  }
                  else {
                    cx = cx + Nx*Ny*step[2];
                    cur[2] = cur[2] +step[2];

                    if (cur[2] > Nz-1 || cur[2] <0){
                      in_domain = 0; 
                    }
                    disMin = tMaxZ - tMax_prev;
                    tMax_prev = tMaxZ;
                    tMaxZ = tMaxZ+tDeltaZ;

                  }
                }

                else {
                  if(tMaxY <tMaxZ){
                    cx = cx + Nx*step[1];
                    cur[1] = cur[1] +step[1];

                    if (cur[1] > Ny-1 || cur[1] < 0){
                      in_domain = 0;  
                    }
                    disMin = tMaxY - tMax_prev;
                    tMax_prev = tMaxY;
                    tMaxY = tMaxY +tDeltaY;
                  }
                  else {
                    cx = cx +Nx*Ny*step[2];
                    cur[2] = cur[2] +step[2];

                    if (cur[2] > Nz-1 || cur[2] < 0){
                      in_domain = 0;  
                    }
                    disMin = tMaxZ - tMax_prev;
                    tMax_prev = tMaxZ;
                    tMaxZ = tMaxZ +tDeltaZ;
                  }
                }
                //this is necessary to find the absorb_coef at the endpoints of each step

                //if (in_domain) this probably is necessary. Dec 20
                ray_location_prev[0] = ray_location[0];
                ray_location_prev[1] = ray_location[1];
                ray_location_prev[2] = ray_location[2];

                ray_location[0] = ray_location[0] + disMin * direction_vector[0];
                ray_location[1] = ray_location[1] + disMin * direction_vector[1];
                ray_location[2] = ray_location[2] + disMin * direction_vector[2];


                /*      //For ray visualization only.
                        if (k==_slice && j==_slice && i==19  ){
                        RayVisX[rRay][qx]=ray_location[0];
                        RayVisY[rRay][qx]=ray_location[1];
                        RayVisZ[rRay][qx]=ray_location[2];
                        qx++;
                        cout << "ray location " << ray_location << endl;

                        }


                //For ray visualization only.
                if (k==_slice && j==_slice &&  i==21 ){
                RayVisX[rRay][qx]=ray_location[0];
                RayVisY[rRay][qx]=ray_location[1];
                RayVisZ[rRay][qx]=ray_location[2];
                qx++;
                cout << "ray location " << ray_location << endl;

                }
                 */
                if (first){
                  chi = absorb_coef[ix]; 
                  first = 0;
                }//end if(first)



                //Because I do these next three lines before the switch statement, I will never have...
                //to worry about cells outside the boundary, or having to decrement opticalthickness or...
                //intensity after I get back inside the domain.
                optical_thickness_prev = optical_thickness;
                optical_thickness += dx_absorb_coef[cx_p]*disMin;


                intensity = intensity*exp(-optical_thickness);//update intensity by Beer's Law

                size++;

                //Eqn 3-15, while accounting for fs. Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
                chi_Iin_cv += chi * (Iout_cv[cx_p] * ( exp(-optical_thickness_prev) - exp(-optical_thickness) ) * fs );////Dec 1
                // Multiply Iin by the current chi, for each ray not just at the end of all the rays.


                /*//visualization only
                  if (i ==21) {
                  if (i_flag){ 
                  cout << endl << endl;
                  i_flag = 0;
                  }
                  }
                  if (k==_slice && j==_slice && (i==19 || i ==21) ){
                  cout  << "chi_Iin_cv " <<  chi_Iin_cv << " absorb_coef " << absorb_coef[cx]  << " cur " << cur[0] << " " << cur[1] << " " << cur[2] << " cx " <<cx << endl;
                  cout << "absorb3D " << absorb_coef_3D[cur[2]][cur[1]][cur[0]] << endl;
                  }
                //end of visualization
                 */


              } //end domain while loop.  ++++++ ++++++++

              if (intensity > d_Threshold){//i.e. if we're doing a reflection

                //puts ray back inside the domain...; 
                intensity*=rho;
                //comment out for cold wall:  Iin_cv += _alpha * Iout_cv[cx] * exp(-optical_thickness)*fs;//!! Right now the temperature of the...
                //boundary is simply the temp of the cell just inside the wall.This is accounting for emission from the walls reacing the origin
                //for non-cold wall, make this a chi_Iin_cv.
                //Comment out for cold, black walls: fs*=rho;//update fs after above Iin reassignment because the reflection is not attenuated by itself.
              }//end reflection if statement

            }//end threshold while loop (ends ray tracing for that ray
            // if (k==_slice && j==_slice && (i==19 || i==21) ){
            // rRay++;
            // }
          }//end N number of rays per cell      

          //Compute Inet.  Iout is blackbody and must be multiplied by absorb_coef. absorb_coef is the kappa of 9.53.
          Inet_cv[ix] = Iout_cv[ix] * absorb_coef[ix] - (chi_Iin_cv/d_NoOfRays); //the last term is from Paula's eqn 3.10
          
          IntVector c(i,j,k);
          fixMe[c] = Inet_cv[ix];

          ix++;//bottom of bottom of cell loop. otherwise ix begins at 1.  
        }// end cell iterator i
      }// end j
    }//end k

    if (_slice > Nx) cout << endl <<"Slice is outside Domain! " << endl << "FIX SLICE!!" << endl;
    if (_slice != Nx/2) cout << endl<<"SLICE IS NOT CENTERED! " << endl;


    FILE *f; //, *fx, *fy, *fz;

    f=fopen("DelDotqA.txt", "w");
    ix = Nx*Ny*_slice;//This can be adjusted to determine which slice of interest is used in visualization.
    for ( j=0; j<Ny; j++){
      for ( i=0; i<Nx; i++){
        //compute flux divergence from Inet and print to file
        fprintf(f, "%lf \t",Inet_cv[ix]*4*_pi);//The last number is 4*pi !! change to have more digits
        ix++;
      }
      fprintf(f, "\n");
    }
    fclose(f);


    //Paula's benchmark 99, for (x,0.5,0) or my (x,39,19) shd be (x,39,(19+20/2) )
    f=fopen("DelDotqline41_50.txt", "w");
    ix = Nx*Ny*_slice + Ny*(Nx/2);//the last term should be Ny*((Nx-1)/2) but because Nx is an int, division by 2 gives the same result
    for (i=0; i<Nx; i++){
      fprintf(f, "%lf \t",Inet_cv[ix]*4*_pi);//The last number is 4*pi !! change to have more digits
      ix++;
    }
    fclose(f);

    /*      //This is a DelDotqline which goes from the bottom to the top through the center.
            f=fopen("DelDotqline_bt.txt", "w");
    //the last term should be Ny*((Nx-1)/2) but because Nx is an int, division by 2 gives the same result
    ix =0;
    for ( k=0; k<Nz; k++){//the indeces of the cells
    for ( j=0; j<Ny; j++){
    for ( i=0; i<Nx; i++){
    if(j==_slice && i==_slice){
    fprintf(f, "%lf \t",Inet_cv[ix]*12.5663706143591729);//The last number is 4*pi !! change to have more digits
    }
    ix++;
    }
    }
    }

    fclose(f);


    //This is a DelDotqline which goes from the south to the north through the center.
    f=fopen("DelDotqline_sn.txt", "w");
    //the last term should be Ny*((Nx-1)/2) but because Nx is an int, division by 2 gives the same result
    ix =0;
    for ( k=0; k<Nz; k++){//the indeces of the cells
    for ( j=0; j<Ny; j++){
    for ( i=0; i<Nx; i++){
    if(k==_slice && i==_slice){
    fprintf(f, "%lf \t",Inet_cv[ix]*12.5663706143591729);//The last number is 4*pi !! change to have more digits
    }
    ix++;
    }
    }
    }

    fclose(f);



     */

    /*      //visualize the rays

            int qx = 0;
            fx=fopen("RayVisX.txt", "w");
            fy=fopen("RayVisY.txt", "w");
            fz=fopen("RayVisZ.txt", "w");

            for(int rr = 0; rr<2*d_NoOfRays;rr++){
            for (qx = 0; qx<Nx*3; qx++){
            fprintf(fx, "%+14.8E  \t",RayVisX[rr][qx]);
            fprintf(fy, "%+14.8E  \t",RayVisY[rr][qx]);
            fprintf(fz, "%+14.8E  \t",RayVisZ[rr][qx]);
            }
            fprintf(fx, "\n");
            fprintf(fy, "\n");
            fprintf(fz, "\n");

            }


            fclose(fx);
            fclose(fy);
            fclose(fz);

     */


    //Calculate the flux divergence which can be saved to the uda file
    // for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    //   c = *iter;
    //   IsaacFlux[c] = Inet_cv[ix]*12.56637;//The last number is 4*pi
    // }




    double end =clock();                        // S_top time the stop watch  
    cout<<endl <<"Used "<<(end-start)*1000/ CLOCKS_PER_SEC<<" milliseconds of CPU time. \n" << endl;// Convert time to ms 

    cout << "Size: " << size << endl;
    double efficiency = size/((end-start)/ CLOCKS_PER_SEC);
    cout << "Efficiency: " << efficiency << " steps per sec" << endl;
  }//end patch loop


} // end ray trace method



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



//Jeremy to do list
//get cell type from dw. X
//get sigmaT^4 rather than T.  Calculate sigmaT^4 before coarsening
//Regression Testing Gold standars
//Change place from where this is called X

