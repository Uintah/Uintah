namespace Uintah{
template <typename T, typename MemSpace, typename RandomGenerator, int m_maxLevels>
struct SlimRayTrace_dataOnion_solveDivQFunctor {

  typedef unsigned long int value_type;
  typedef typename RandomGenerator::generator_type rnd_type;

  LevelParamsML                          m_levelParamsML[m_maxLevels];
  double                                 m_domain_BB_Lo[3];
  double                                 m_domain_BB_Hi[3];
  int                                    m_fineLevel_ROI_Lo[3];
  int                                    m_fineLevel_ROI_Hi[3];
  KokkosView3<const double, MemSpace> m_abskgSigmaT4CellType[m_maxLevels];
  KokkosView3<double, MemSpace>       m_divQ_fine;
  KokkosView3<double, MemSpace>       m_radiationVolq_fine;
  double                                 m_d_threshold;
  bool                                   m_d_allowReflect;
  int                                    m_d_nDivQRays;
  bool                                   m_d_CCRays;
  int                                    m_halo;
  RandomGenerator                        m_rand_pool;

  SlimRayTrace_dataOnion_solveDivQFunctor( LevelParamsML                        levelParamsML[m_maxLevels]
                                     , double                                   domain_BB_Lo[3]
                                     , double                                   domain_BB_Hi[3]
                                     , int                                      fineLevel_ROI_Lo[3]
                                     , int                                      fineLevel_ROI_Hi[3]
                                     , KokkosView3<const double, MemSpace>   abskgSigmaT4CellType[m_maxLevels]
                                     , KokkosView3<double, MemSpace>       & divQ_fine
                                     , KokkosView3<double, MemSpace>       & radiationVolq_fine
                                     , double                                 & d_threshold
                                     , bool                                   & d_allowReflect
                                     , int                                    & d_nDivQRays
                                     , bool                                   & d_CCRays
                                     , int                                      halo=9999 // defaults to patch based functionality, but less efficient  = /
                                     )
    : m_divQ_fine          ( divQ_fine )
    , m_radiationVolq_fine ( radiationVolq_fine )
    , m_d_threshold        ( d_threshold )
    , m_d_allowReflect     ( d_allowReflect )
    , m_d_nDivQRays        ( d_nDivQRays )
    , m_d_CCRays           ( d_CCRays )
    , m_halo               ( halo )
  {
    for ( int L = 0; L < m_maxLevels; L++ ) {

      m_levelParamsML[L] = levelParamsML[L];

      m_abskgSigmaT4CellType[L] = abskgSigmaT4CellType[L];

    }

    m_domain_BB_Lo[0] = domain_BB_Lo[0];
    m_domain_BB_Lo[1] = domain_BB_Lo[1];
    m_domain_BB_Lo[2] = domain_BB_Lo[2];

    m_domain_BB_Hi[0] = domain_BB_Hi[0];
    m_domain_BB_Hi[1] = domain_BB_Hi[1];
    m_domain_BB_Hi[2] = domain_BB_Hi[2];

    m_fineLevel_ROI_Lo[0] = fineLevel_ROI_Lo[0];
    m_fineLevel_ROI_Lo[1] = fineLevel_ROI_Lo[1];
    m_fineLevel_ROI_Lo[2] = fineLevel_ROI_Lo[2];

    m_fineLevel_ROI_Hi[0] = fineLevel_ROI_Hi[0];
    m_fineLevel_ROI_Hi[1] = fineLevel_ROI_Hi[1];
    m_fineLevel_ROI_Hi[2] = fineLevel_ROI_Hi[2];

#ifndef FIXED_RANDOM_NUM
    KokkosRandom<RandomGenerator> kokkosRand( true );
    m_rand_pool = kokkosRand.getRandPool();
#endif
  }

  // This operator() replaces the cellIterator loop used to solve DivQ
  KOKKOS_INLINE_FUNCTION
  void operator() ( const int i, const int j, const int k, value_type & m_nRaySteps ) const {

#ifndef FIXED_RANDOM_NUM
    // Each thread needs a unique state
    rnd_type rand_gen = m_rand_pool.get_state();
#endif

    //____________________________________________________________________________________________//
    //==== START for (CellIterator iter = finePatch->getCellIterator(); !iter.done(); iter++) ====//

    int L = m_maxLevels - 1;


    double sumI = 0;

    double rayOrigin[3];
    int ijk[3] = {i,j,k};
    m_levelParamsML[L].getCellPosition(ijk, rayOrigin);

    const Combined_RMCRT_Required_Vars abskgSigmaT4CellTypeConst = reinterpret_cast<const Combined_RMCRT_Required_Vars&>(m_abskgSigmaT4CellType[m_maxLevels-1](i,j,k));
    //__________________________________
    // A negative abskg value indicates a wall, a postive abskg value indicates the ray can move through it
    // Floats and doubles allow for -0.0f and +0.0f values, they are different, and this can be detected through
    // the std::signbit tool, but that isn't GPU portable, so instead just look at the first bit.

    if ( reinterpret_cast<const int&>(abskgSigmaT4CellTypeConst.abskg) & 0x80000000 ) {
       return; // No radiation in intrusions
    }
     // --------------------------------create cell extent ----------------------------------//
     //--  This computes a cell extent for each cell origin, using a constant extent range --//
     // -------------------------------------------------------------------------------------//
    int  local_region[m_maxLevels][2][3];
    if (m_halo>9998){
      for( int iLev=1; iLev < m_maxLevels; iLev++){
        for (int isign=0; isign <2 ; isign++){
          for (int idir=0; idir <3 ; idir++){
            local_region[iLev][isign][idir]=isign ?  m_levelParamsML[iLev].regionHi[idir] : m_levelParamsML[iLev].regionLo[idir];
          }
        }
      }
    }else{
      // set up cell based ROI
      int local_cur[3]      = {i, j, k };
      for( int iLev=0; iLev < m_maxLevels-1; iLev++){
        for (int idir=0; idir <3 ; idir++){
          local_region[m_maxLevels-iLev-1][0][idir]=local_cur[idir] +  -m_halo ;
          local_region[m_maxLevels-iLev-1][1][idir]=local_cur[idir] +   m_halo+1 ;

        }
        m_levelParamsML[m_maxLevels-1-iLev].mapCellToCoarser(local_cur);
      }
      // use same ROI as patch based for coarsest level
      for (int isign=0; isign <2 ; isign++){
        for (int idir=0; idir <3 ; idir++){
          local_region[0][isign][idir]=isign ?  m_levelParamsML[0].regionHi[idir] : m_levelParamsML[0].regionLo[idir];
        }
      }
    }
     // -------------------------------------------------------------------------------------//
     // ----------------------------- end create cell extent --------------------------------//
     // -------------------------------------------------------------------------------------//

    int dir = -9; // Hard-coded for NONE
    //  Ray loop
    for ( int iRay = 0; iRay < m_d_nDivQRays; iRay++ ) {

      //________________________________________________________//
      //==== START findRayDirection(mTwister, origin, iRay) ====//

      // Random Points On Sphere
#ifdef FIXED_RANDOM_NUM
      double plusMinus_one = 2.0 * 0.3 - 1.0 + DBL_EPSILON;   // Add fuzz to avoid inf in 1/dirVector
      double r = sqrt( 1.0 - plusMinus_one * plusMinus_one ); // Radius of circle at z
      double theta = 2.0 * M_PI * 0.3;                        // Uniform betwen 0-2Pi
#else
      double plusMinus_one = 2.0 * Kokkos::rand<rnd_type, double>::draw(rand_gen) - 1.0 + DBL_EPSILON; // Add fuzz to avoid inf in 1/dirVector
      double r = sqrt( 1.0 - plusMinus_one * plusMinus_one );                                          // Radius of circle at z
      double theta = 2.0 * M_PI * Kokkos::rand<rnd_type, double>::draw(rand_gen);                      // Uniform betwen 0-2Pi
#endif

      const double direction_vector[3]={ r * cos( theta ),r * sin( theta ),plusMinus_one} ;

//`==========DEBUGGING==========//
#if ( FIXED_RAY_DIR == 1)
      direction_vector[0] = 0.707106781186548 * SIGN;
      direction_vector[1] = 0.707106781186548 * SIGN;
      direction_vector[2] = 0.0               * SIGN;
#elif ( FIXED_RAY_DIR == 2 )
      direction_vector[0] = 0.707106781186548 * SIGN;
      direction_vector[1] = 0.0               * SIGN;
      direction_vector[2] = 0.707106781186548 * SIGN;
#elif ( FIXED_RAY_DIR == 3 )
      direction_vector[0] = 0.0               * SIGN;
      direction_vector[1] = 0.707106781186548 * SIGN;
      direction_vector[2] = 0.707106781186548 * SIGN;
#elif ( FIXED_RAY_DIR == 4 )
      direction_vector[0] = 0.707106781186548 * SIGN;
      direction_vector[1] = 0.707106781186548 * SIGN;
      direction_vector[2] = 0.707106781186548 * SIGN;
#elif ( FIXED_RAY_DIR == 5 )
      direction_vector[0] = 1 * SIGN;
      direction_vector[1] = 0 * SIGN;
      direction_vector[2] = 0 * SIGN;
#elif ( FIXED_RAY_DIR == 6 )
      direction_vector[0] = 0 * SIGN;
      direction_vector[1] = 1 * SIGN;
      direction_vector[2] = 0 * SIGN;
#elif ( FIXED_RAY_DIR == 7 )
      direction_vector[0] = 0 * SIGN;
      direction_vector[1] = 0 * SIGN;
      direction_vector[2] = 1 * SIGN;
#else
#endif
//===========DEBUGGING==========`//

      //______________________________________________________//
      //==== END findRayDirection(mTwister, origin, iRay) ====//


      //___________________________________________________________________________//
      //==== START ray_Origin(mTwister, CC_pos, Dx[my_L], d_CCRays, rayOrigin) ====//

      //_______________________________________//
      //==== START updateSumI_ML< T >(...) ====//

      int L       = m_maxLevels - 1;  // finest level  // dynamic

      int cur[3]      = { i, j, k };                  //dynamic

      const double inv_direction[3] = { 1.0 / direction_vector[0]
                                , 1.0 / direction_vector[1]
                                , 1.0 / direction_vector[2] }; //const

//`==========TESTING==========//
#if DEBUG == 1
      if ( isDbgCell(i,j,k) ) {
        printf( "        updateSumI_ML: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n",
                i, j, k, direction_vector[0], direction_vector[1], direction_vector[2], rayOrigin[0], rayOrigin[1], rayOrigin[2] );
      }
#endif
//===========TESTING==========`//

      //______________________________________________________//
      //==== START raySignStep(sign, step, ray_direction) ====//

        const int step[3]   = { direction_vector[0] > 0.0 ? 1 : -1   ,direction_vector[1] > 0.0 ? 1 : -1,  direction_vector[2] > 0.0 ? 1 : -1 };

      //____________________________________________________//
      //==== END raySignStep(sign, step, ray_direction) ====//


      // tMax is the physical distance from the ray origin to each of the respective planes of intersection
      double tMaxV[3];
      tMaxV[0] = 0.5*((double) step[0] ) * m_levelParamsML[L].Dx[0] * inv_direction[0]; // signs cancle so always positive
      tMaxV[1] = 0.5*((double) step[1] ) * m_levelParamsML[L].Dx[1] * inv_direction[1];
      tMaxV[2] = 0.5*((double) step[2] ) * m_levelParamsML[L].Dx[2] * inv_direction[2];

      double tDelta[m_maxLevels][3];
      for ( int Lev = m_maxLevels - 1; Lev > -1; Lev-- ) {
        //Length of t to traverse one cell
        tDelta[Lev][0] = fabs( inv_direction[0] ) * m_levelParamsML[Lev].Dx[0];
        tDelta[Lev][1] = fabs( inv_direction[1] ) * m_levelParamsML[Lev].Dx[1];
        tDelta[Lev][2] = fabs( inv_direction[2] ) * m_levelParamsML[Lev].Dx[2];
      }

      //Initializes the following values for each ray
      double old_length           = 0.0;
      double expOpticalThick_prev = 1.0;         // exp(-opticalThick_prev)
      double CC_pos[3]            = { rayOrigin[0], rayOrigin[1], rayOrigin[2] };

      //______________________________________________________________________
      //  Threshold  loop
      Combined_RMCRT_Required_Vars curAbskgSigmaT4CellType = abskgSigmaT4CellTypeConst;
      //reinterpret_cast<const Combined_RMCRT_Required_Vars&>(m_abskgSigmaT4CellType[L](cur[0], cur[1], cur[2]));

      printf("In cell (%d, %d, %d). Ray #%d is at (%d,%d,%d) (init 0)\n", i,j,k, iRay, cur[0], cur[1], cur[2]);
      // Move the ray ahead.  We know we can't drop a coarse level yet (halos must be at least 1),
      // The cur/ray can't go out of bounds, because at worst this moves the ray into a wall.
      dir = tMaxV[0] < tMaxV[1] ? (tMaxV[0] < tMaxV[2] ? 0 : 2) : (tMaxV[1] < tMaxV[2] ? 1 : 2);
      cur[dir]  +=  step[dir];
      printf("In cell (%d, %d, %d). Ray #%d is at (%d,%d,%d) (init 1)\n", i,j,k, iRay, cur[0], cur[1], cur[2]);
      double distanceTraveled = ( tMaxV[dir] - old_length );
      old_length  = tMaxV[dir];
      tMaxV[dir] = tMaxV[dir] + tDelta[L][dir];

      //Warning - trickiness ahead: The ray started in cell 1, and previously confirmed it is not in a wall.
      //The ray has moved into cell 2, but the no values have been obtained yet for this new cell 2
      //This new cell 2 could be a wall, but it can't be at a point where we would coarsen yet (assuming at least 1 halo cell layer)
      //Start the loop, get the values at cell 2, and move the ray to a position ready to get cell 3.  We may coarsen after cur is moved into cell 3.
      do {
        dir = tMaxV[0] < tMaxV[1] ? (tMaxV[0] < tMaxV[2] ? 0 : 2) : (tMaxV[1] < tMaxV[2] ? 1 : 2);

        double distanceTraveled = ( tMaxV[dir] - old_length );
        double expOpticalThick = (1. - curAbskgSigmaT4CellType.abskg * distanceTraveled )*expOpticalThick_prev; // exp approximation
        sumI += curAbskgSigmaT4CellType.sigmaT4 * ( expOpticalThick_prev - expOpticalThick ) ;
        expOpticalThick_prev = expOpticalThick;

        cur[dir]  += step[dir];
        printf("In cell (%d, %d, %d). Ray #%d is at (%d,%d,%d) (loop)\n", i,j,k, iRay, cur[0], cur[1], cur[2]);
        old_length              = tMaxV[dir];
        //__________________________________
        // When moving to a coarse level tmax will change only in the direction the ray is moving
        if ( local_region[L][0][dir] > cur[dir] ||  local_region[L][1][dir] <= cur[dir] ){ // only one of these comparisons is needed, based on the sign of the current direction, consider placing in a container so that high/low can be accessed via index.  possibly a 0.2% speedup
          m_levelParamsML[L].mapCellToCoarser(cur);
          L--;  // move to a coarser level
          m_levelParamsML[L].getCellPosition(cur, CC_pos);
          printf("In cell (%d, %d, %d), Dropped a level to level %d. Ray #%d is at (%d,%d,%d)\n", i,j,k, L, iRay, cur[0], cur[1], cur[2]);
          double rayDx_Level =rayOrigin[dir]+tMaxV[dir]*direction_vector[dir]  - ( CC_pos[dir] - 0.5 * m_levelParamsML[L].Dx[dir] ); // account for dropping levels in middle of cell, could remove if ROIs ensured droping down on fine-grid/coarse-grid interface
          double tMax_tmp    = ( std::max((double) step[dir],0.0) * m_levelParamsML[L].Dx[dir] - rayDx_Level ) * inv_direction[dir];
          tMaxV[dir]         += tMax_tmp;
        } else {
          tMaxV[dir]         = tMaxV[dir] + tDelta[L][dir];
        }
        curAbskgSigmaT4CellType = reinterpret_cast<const Combined_RMCRT_Required_Vars&>(m_abskgSigmaT4CellType[L](cur[0], cur[1], cur[2]));
        printf("In cell (%d, %d, %d). Ray #%d, checking the value of abskg at (%d,%d,%d) to see if we are at a wall: %g\n", i,j,k, iRay, cur[0], cur[1], cur[2], curAbskgSigmaT4CellType.abskg);
      } while ( ! (reinterpret_cast<int&>(curAbskgSigmaT4CellType.abskg) & 0x80000000) );// end domain while loop
      T wallEmissivity = ( fabs(curAbskgSigmaT4CellType.abskg) > 1.0 ) ? 1.0 : fabs(curAbskgSigmaT4CellType.abskg);  // Ensure wall emissivity doesn't exceed one
      sumI += wallEmissivity * curAbskgSigmaT4CellType.sigmaT4 * expOpticalThick_prev;
    }  // end ray loop

    //__________________________________
    //  Compute divQ
    //m_divQ_fine(i,j,k) = -4.0 * M_PI * m_abskg[fine_L](i,j,k) * ( m_sigmaT4OverPi[fine_L](i,j,k) - ( sumI / m_d_nDivQRays) );
    m_divQ_fine(i,j,k) = -4.0 * M_PI * abskgSigmaT4CellTypeConst.abskg * ( abskgSigmaT4CellTypeConst.sigmaT4 - ( sumI / m_d_nDivQRays) );

    // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used
    m_radiationVolq_fine(i,j,k) = 4.0 * M_PI * ( sumI / m_d_nDivQRays );

//`==========TESTING==========//
#if DEBUG == 1
    if ( isDbgCell(i,j,k) ) {
      printf( "\n      [%d, %d, %d]  sumI: %g  divQ: %g radiationVolq: %g  abskg: %g,    sigmaT4: %g \n\n",
              i, j, k, sumI, m_divQ_fine(i,j,k), m_radiationVolq_fine(i,j,k), m_abskg[m_maxLevels-1](i,j,k), m_sigmaT4OverPi[m_maxLevels-1](i,j,k) );
    }
#endif
/*===========TESTING==========`*/

#ifndef FIXED_RANDOM_NUM
    m_rand_pool.free_state(rand_gen);
#endif

  }  // end operator()
};   // end SlimRayTrace_dataOnion_solveDivQFunctor
}
