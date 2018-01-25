template <typename T, typename RandomGenerator, int m_maxLevels>
struct SlimRayTrace_dataOnion_solveDivQFunctor {

  typedef unsigned long int value_type;
  typedef typename RandomGenerator::generator_type rnd_type;

  LevelParamsML                       m_levelParamsML[m_maxLevels];
  RMCRT_flags                         m_RT_flags;
  double                              m_domain_BB_Lo[3];
  double                              m_domain_BB_Hi[3];
  int                                 m_fineLevel_ROI_Lo[3];
  int                                 m_fineLevel_ROI_Hi[3];
  KokkosView3<const double>                 m_abskgSigmaT4CellType[m_maxLevels];
  KokkosView3<double>                 m_divQ_fine;
  KokkosView3<double>                 m_radiationVolq_fine;
  double                              m_d_threshold;
  bool                                m_d_allowReflect;
  int                                 m_d_nDivQRays;
  bool                                m_d_CCRays;
  RandomGenerator                     m_rand_pool;

  SlimRayTrace_dataOnion_solveDivQFunctor( LevelParamsML                       levelParamsML[m_maxLevels]
                                     , RMCRT_flags                       & RT_flags
                                     , double                              domain_BB_Lo[3]
                                     , double                              domain_BB_Hi[3]
                                     , int                                 fineLevel_ROI_Lo[3]
                                     , int                                 fineLevel_ROI_Hi[3]
                                     , KokkosView3<const double>         abskgSigmaT4CellType[m_maxLevels]
                                     , KokkosView3<double>               & divQ_fine
                                     , KokkosView3<double>               & radiationVolq_fine
                                     , double                            & d_threshold
                                     , bool                              & d_allowReflect
                                     , int                               & d_nDivQRays
                                     , bool                              & d_CCRays
                                     )
    : m_RT_flags           ( RT_flags )
    , m_divQ_fine          ( divQ_fine )
    , m_radiationVolq_fine ( radiationVolq_fine )
    , m_d_threshold        ( d_threshold )
    , m_d_allowReflect     ( d_allowReflect )
    , m_d_nDivQRays        ( d_nDivQRays )
    , m_d_CCRays           ( d_CCRays )
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
  void operator() ( int n, unsigned long int & m_nRaySteps ) const {

#ifndef FIXED_RANDOM_NUM
    // Each thread needs a unique state
    rnd_type rand_gen = m_rand_pool.get_state();
#endif

    //____________________________________________________________________________________________//
    //==== START for (CellIterator iter = finePatch->getCellIterator(); !iter.done(); iter++) ====//

    int L = m_maxLevels - 1;

    int i, j, k;
    unsigned int threadID = n + m_RT_flags.startCell;
    i = (threadID % m_RT_flags.finePatchSize.x) + m_RT_flags.finePatchLow.x;
    j = ((threadID % (m_RT_flags.finePatchSize.x * m_RT_flags.finePatchSize.y)) / (m_RT_flags.finePatchSize.x)) + m_RT_flags.finePatchLow.y;
    k = (threadID / (m_RT_flags.finePatchSize.x * m_RT_flags.finePatchSize.y)) + m_RT_flags.finePatchLow.z;

    //while (false) {
    while ( threadID < m_RT_flags.endCell ) {

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

        //if ( m_d_CCRays == false ) {

//#ifdef FIXED_RANDOM_NUM
          //double x = 0.3 * m_levelParamsML[my_L].Dx[0];
          //double y = 0.3 * m_levelParamsML[my_L].Dx[1];
          //double z = 0.3 * m_levelParamsML[my_L].Dx[2];
//#else
          //double x = Kokkos::rand<rnd_type, double>::draw(rand_gen) * m_levelParamsML[my_L].Dx[0];
          //double y = Kokkos::rand<rnd_type, double>::draw(rand_gen) * m_levelParamsML[my_L].Dx[1];
          //double z = Kokkos::rand<rnd_type, double>::draw(rand_gen) * m_levelParamsML[my_L].Dx[2];
//#endif

          //double offset[3] = { x, y, z };  // Note you HAVE to compute the components separately to ensure that the
                                           //  random numbers called in the x,y,z order - Todd

          //if ( offset[0] > m_levelParamsML[my_L].Dx[0] ||
               //offset[1] > m_levelParamsML[my_L].Dx[1] ||
               //offset[2] > m_levelParamsML[my_L].Dx[2] ) {
            //printf(" Warning:ray_Origin  The Kokkos random number generator has returned garbage (%g, %g, %g) Now forcing the ray origin to be located at the cell-center\n",
                //offset[0], offset[1], offset[2]);
            //offset[0] = 0.5 * m_levelParamsML[my_L].Dx[0];
            //offset[1] = 0.5 * m_levelParamsML[my_L].Dx[1];
            //offset[2] = 0.5 * m_levelParamsML[my_L].Dx[2];
          //}

          //rayOrigin[0] =  CC_pos[0] - 0.5 * m_levelParamsML[my_L].Dx[0] + offset[0];
          //rayOrigin[1] =  CC_pos[1] - 0.5 * m_levelParamsML[my_L].Dx[1] + offset[1];
          //rayOrigin[2] =  CC_pos[2] - 0.5 * m_levelParamsML[my_L].Dx[2] + offset[2];
        //}
        //else {
          //rayOrigin[0] = CC_pos[0];
          //rayOrigin[1] = CC_pos[1];
          //rayOrigin[2] = CC_pos[2];
        //}

        //___________________________________________________________________//
        //==== END ray_Origin(mTwister, CC_pos, Dx, d_CCRays, rayOrigin) ====//

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

        //__________________________________
        // Define tMax & tDelta on all levels
        // Go from finest to coarsest level so you can compare
        // with 1L rayTrace results.
        //double CC_posOrigin[3];
        //m_levelParamsML[L].getCellPosition(ijk, CC_posOrigin);



        //double CC_posOrigin[3] = { m_fineLevel->getAnchor().x() + ( m_Dx[L][0] * i ) + ( 0.5 * m_Dx[L][0] )
        //                         , m_fineLevel->getAnchor().y() + ( m_Dx[L][1] * j ) + ( 0.5 * m_Dx[L][1] )
        //                         , m_fineLevel->getAnchor().z() + ( m_Dx[L][2] * k ) + ( 0.5 * m_Dx[L][2] ) };

        // rayDx is the distance from bottom, left, back, corner of cell to ray
        //double rayDx[3];

        // tMax is the physical distance from the ray origin to each of the respective planes of intersection
        double tMaxV[3];
        //tMaxV[0] = ( std::max((double) step[0] ,0.0)* m_levelParamsML[L].Dx[0] - rayDx[0] ) * inv_direction[0];
        //tMaxV[1] = ( std::max((double) step[1] ,0.0)* m_levelParamsML[L].Dx[1] - rayDx[1] ) * inv_direction[1];
        //tMaxV[2] = ( std::max((double) step[2] ,0.0)* m_levelParamsML[L].Dx[2] - rayDx[2] ) * inv_direction[2];
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
        double rayLength            = 0.0;
        double CC_pos[3]            = { rayOrigin[0], rayOrigin[1], rayOrigin[2] };

        //______________________________________________________________________
        //  Threshold  loop

        // The starting cell was previously checked and ensured the ray at cur isn't in a wall.  Obtain a copy of it.
        Combined_RMCRT_Required_Vars prevAbskgSigmaT4CellType;
        Combined_RMCRT_Required_Vars curAbskgSigmaT4CellType = abskgSigmaT4CellTypeConst;
        //reinterpret_cast<const Combined_RMCRT_Required_Vars&>(m_abskgSigmaT4CellType[L](cur[0], cur[1], cur[2]));

        // Move the ray ahead.  We know we can't drop a coarse level yet (halos must be at least 1),
        // The cur/ray can't go out of bounds, because at worst this moves the ray into a wall.
        dir = tMaxV[0] < tMaxV[1] ? (tMaxV[0] < tMaxV[2] ? 0 : 2) : (tMaxV[1] < tMaxV[2] ? 1 : 2);
        cur[dir]  +=  step[dir];
        double distanceTraveled = ( tMaxV[dir] - old_length );
        old_length  = tMaxV[dir];
        //rayLength += distanceTraveled;
        tMaxV[dir] = tMaxV[dir] + tDelta[L][dir];

        //The ray started in cell 1, and previously confirmed its not in a wall.
        //The ray has moved into cell 2, but the no values have been obtained yet for this new cell 2
        //This new cell 2 could be a wall, but it can't be at a point where we would coarsen yet (assuming at least 1 halo cell layer)
        //Start the loop, get the values at cell 2, and move the ray to a position ready to get cell 3.  We may coarsen after cur is moved into cell 3.
        do {

          // Immediately prepare the next value by pipelining it.
          prevAbskgSigmaT4CellType = curAbskgSigmaT4CellType;
          curAbskgSigmaT4CellType = reinterpret_cast<const Combined_RMCRT_Required_Vars&>(m_abskgSigmaT4CellType[L](cur[0], cur[1], cur[2]));

          // Do some updated computations
          double expOpticalThick = (1. - prevAbskgSigmaT4CellType.abskg * distanceTraveled )*expOpticalThick_prev; // exp approximation
          sumI += prevAbskgSigmaT4CellType.sigmaT4 * ( expOpticalThick_prev - expOpticalThick ) ;
          expOpticalThick_prev = expOpticalThick;

          // Determine which cell the ray will enter next
          // Note, cur may already be at a wall, so this may update cur and put it through/past a wall.
          // It may be needed to use a bounding box check here...
          dir = tMaxV[0] < tMaxV[1] ? (tMaxV[0] < tMaxV[2] ? 0 : 2) : (tMaxV[1] < tMaxV[2] ? 1 : 2);
          cur[dir]  += step[dir];

          //__________________________________
          //  Update marching variables
          distanceTraveled = ( tMaxV[dir] - old_length );
          old_length              = tMaxV[dir];
          //rayLength         += distanceTraveled;
          //__________________________________
          // When moving to a coarse level tmax will change only in the direction the ray is moving
          if ( m_levelParamsML[L].regionLo[dir] > cur[dir] || m_levelParamsML[L].regionHi[dir] <= cur[dir] ){
            if (L == 0 ) {
              // The ray is heading out of the region, and it is also already on the coarsest level.
              // It is possible the cur is now 1) already in a wall, or 2) past the wall.
              // To see if it is #2, past the wall, the prior value can be checked.
              if (!(reinterpret_cast<int&>(curAbskgSigmaT4CellType.abskg) & 0x80000000) ) {
              //if ( curAbskgSigmaT4CellType.abskg > 0 ) {
                // The prior value wasn't in a wall, but we assume cur is in a wall.
                // So update the values.
                double expOpticalThick = (1. - curAbskgSigmaT4CellType.abskg * distanceTraveled )*expOpticalThick_prev; // exp approximation
                sumI += curAbskgSigmaT4CellType.sigmaT4 * ( expOpticalThick_prev - expOpticalThick ) ;
                expOpticalThick_prev = expOpticalThick;

                // Read out one last value for computations after the loop?  Is this needed?
                curAbskgSigmaT4CellType = reinterpret_cast<const Combined_RMCRT_Required_Vars&>(m_abskgSigmaT4CellType[L](cur[0], cur[1], cur[2]));
              }

              break;
            }
            m_levelParamsML[L].mapCellToCoarser(cur);
            L--;  // move to a coarser level
            m_levelParamsML[L].getCellPosition(cur, CC_pos);

            double rayDx_Level = rayOrigin[dir] + distanceTraveled*direction_vector[dir] - ( CC_pos[dir] - 0.5 * m_levelParamsML[L].Dx[dir] );
            double tMax_tmp    = ( std::max((double) step[dir],0.0) * m_levelParamsML[L].Dx[dir] - rayDx_Level ) * inv_direction[dir];
            tMaxV[dir]         += tMax_tmp;
          } else {
            tMaxV[dir]         = tMaxV[dir] + tDelta[L][dir];
          }

        //} while ( curAbskgSigmaT4CellType.abskg > 0 );// end domain while loop 
        } while ( ! (reinterpret_cast<int&>(curAbskgSigmaT4CellType.abskg) & 0x80000000) );// end domain while loop 
        //TODO: Turn back on fabs 
        T wallEmissivity = ( curAbskgSigmaT4CellType.abskg > 1.0 ) ? 1.0 : curAbskgSigmaT4CellType.abskg;  // Ensure wall emissivity doesn't exceed one
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
        threadID += (m_RT_flags.cellsPerGroup);
        i = (threadID % m_RT_flags.finePatchSize.x) + m_RT_flags.finePatchLow.x;
        j = ((threadID % (m_RT_flags.finePatchSize.x * m_RT_flags.finePatchSize.y)) / (m_RT_flags.finePatchSize.x)) + m_RT_flags.finePatchLow.y;
        k = (threadID / (m_RT_flags.finePatchSize.x * m_RT_flags.finePatchSize.y)) + m_RT_flags.finePatchLow.z;
      } // end while (threadID < RT_flags.endCell) {
#ifndef FIXED_RANDOM_NUM
    m_rand_pool.free_state(rand_gen);
#endif

  }  // end operator()
};   // end SlimRayTrace_dataOnion_solveDivQFunctor
