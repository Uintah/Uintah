/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_planeAverage_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_planeAverage_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <map>
#include <vector>

namespace Uintah {


/*______________________________________________________________________

GENERAL INFORMATION

   planeAverage.h

   This computes the spatial average over a plane in the domain

   Todd Harman
   Department of Mechanical Engineering
   University of Utah
______________________________________________________________________*/



//______________________________________________________________________

  class planeAverage : public AnalysisModule {
  public:

    planeAverage(const ProcessorGroup   * myworld,
                 const MaterialManagerP   materialManager,
                 const ProblemSpecP     & module_spec);
    planeAverage();

    virtual ~planeAverage();

    virtual void problemSetup(const ProblemSpecP  & prob_spec,
                              const ProblemSpecP  & restart_prob_spec,
                              GridP               & grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleRestartInitialize(SchedulerP   & sched,
                                           const LevelP & level);

    virtual void restartInitialize();

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                             const LevelP & level) {};

    enum weightingType { NCELLS, MASS, NONE };

  private:

    //__________________________________
    //  This is a wrapper to create a vector of objects of different types(planarVar)
    struct planarVarBase{

      //___________________________________
      //  MPI user defined function for computing sum for Uintah::Point
      //  we need this so each plane location has a non-zero value
      static void plusEqualPoint( Point * in,
                                  Point * inOut,
                                  int   * len,
                                  MPI_Datatype * type)
      {
        for (auto i=0; i< *len; i++){
          inOut[i].x( inOut[i].x() + in[i].x() );
          inOut[i].y( inOut[i].y() + in[i].y() );
          inOut[i].z( inOut[i].z() + in[i].z() );
        }
      }

      public:
        VarLabel* label;
        int matl;
        int level;
        int nPlanes;                       // number of avg planes
        const int rootRank = 0;
        std::vector<Point>  CC_pos;        // cell center position
        std::vector<double> weight;        // weighting to compute ave
        std::vector<int>    nCells;        // number of cells per plane.  Each plane could differ with AMR
        weightingType       weightType;

        TypeDescription::Type baseType;
        TypeDescription::Type subType;

        //__________________________________
        void set_nPlanes(const int in) { nPlanes = in; }
        int  get_nPlanes() { return nPlanes; }

        //__________________________________
        void getPlanarWeight( std::vector<double> & a,
                              std::vector<int>    & b )
        {
          a = weight;
          b = nCells;
        }
        //__________________________________
        void setPlanarWeight( std::vector<double> & a,
                              std::vector<int>    & b )
        {
          weight = a;
          nCells = b;
        }

        //__________________________________
        // sum over all procs the weight and nPlanar cells
        void ReduceWeight( const int rank )
        {
          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &weight.front(), nPlanes, MPI_DOUBLE, MPI_SUM, rootRank, MPI_COMM_WORLD);
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &nCells.front(), nPlanes, MPI_INT,    MPI_SUM, rootRank, MPI_COMM_WORLD);
          } else {
            Uintah::MPI::Reduce(  &weight.front(), 0,            nPlanes, MPI_DOUBLE, MPI_SUM, rootRank, MPI_COMM_WORLD);
            Uintah::MPI::Reduce(  &nCells.front(), 0,            nPlanes, MPI_INT,    MPI_SUM, rootRank, MPI_COMM_WORLD);
          }
        }

        //__________________________________
        void setCC_pos( std::vector<Point>  & pos,
                        const unsigned lo,
                        const unsigned hi)
        {
          for ( auto z = lo; z<hi; z++ ) {
            CC_pos[z] = pos[z];
          }
        }

        //__________________________________
        void ReduceCC_pos( const int rank )
        {
          MPI_Datatype  mpitype;
          Uintah::MPI::Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
          Uintah::MPI::Type_commit( &mpitype );

          MPI_Op point_add;
          MPI_Op_create( (MPI_User_function*) plusEqualPoint, 1, &point_add );

          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &CC_pos.front(), nPlanes, mpitype, point_add, rootRank, MPI_COMM_WORLD );
          } else {
            Uintah::MPI::Reduce(  &CC_pos.front(), 0,            nPlanes, mpitype, point_add, rootRank, MPI_COMM_WORLD );
          }
          MPI_Op_free( &point_add );
        }

        //__________________________________
        //   VIRTUAL FUNCTIONS

        virtual void reserve() = 0;

        // virtual templated functions are not allowed in C++11
        // instantiate the various types
        virtual  void getPlanarSum( std::vector<double>& ave ){}
        virtual  void getPlanarSum( std::vector<Vector>& ave ){}

        virtual  void setPlanarSum( std::vector<double> & ave ){}
        virtual  void setPlanarSum( std::vector<Vector> & ave ){}

        virtual  void zero_all_vars(){}

        virtual  void ReduceVar(const int rank ) = 0;

        virtual  void printAverage( FILE* & fp,
                                    const int levelIndex,
                                    const double simTime ) = 0;
    };

    //  It's simple and straight forward to use a double and vector class
    //  A templated class would be ideal but that involves too much C++ magic
    //
    //______________________________________________________________________
    //  Class that holds the planar quantities     DOUBLE
    class planarVar_double: public planarVarBase{

      private:
        std::vector<double> sum;
      public:

        //__________________________________
        void reserve()
        {
          CC_pos.resize(nPlanes, Point(0.,0.,0.));
          sum.resize(   nPlanes, 0.);
          weight.resize(nPlanes, 0.);
          nCells.resize(nPlanes, 0 );
        }

        //__________________________________
        void getPlanarSum( std::vector<double> & me ) { me = sum; }
        void setPlanarSum( std::vector<double> & me ) { sum = me; }

        //__________________________________
        void zero_all_vars()
        {
          for(unsigned i=0; i<sum.size(); i++ ){
            CC_pos[i] = Point(0,0,0);
            sum[i]    = 0.0;
            weight[i] = 0.0;
            nCells[i] = 0;
          }
        }

        //__________________________________
        void ReduceVar( const int rank )
        {
          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &sum.front(), nPlanes, MPI_DOUBLE, MPI_SUM, rootRank, MPI_COMM_WORLD);
          } else {
            Uintah::MPI::Reduce(  &sum.front(), 0,            nPlanes, MPI_DOUBLE, MPI_SUM, rootRank, MPI_COMM_WORLD);
          }
        }

        //__________________________________
        void printAverage( FILE* & fp,
                           const int levelIndex,
                           const double simTime )
        {
          fprintf( fp,"# Level: %i \n", levelIndex );
          fprintf( fp,"# Simulation time: %16.15E \n", simTime );
          fprintf( fp,"# Plane location x,y,z           Average\n" );
          fprintf( fp,"#________CC_loc.x_______________CC_loc.y______________CC_loc.z_______________ave" );
          if( weightType == NCELLS ){
            fprintf( fp,"___________nCells\n" );
          }
          else if( weightType == MASS ){
            fprintf( fp,"___________weight___________\n" );
          }

          // loop over each plane, compute the ave and write to file
          for ( unsigned i =0; i< sum.size(); i++ ){

            fprintf( fp, "%16.15E  %16.15E  %16.15E ", CC_pos[i].x(), CC_pos[i].y(), CC_pos[i].z() );
            
            if( weightType == NCELLS ){
              double avg = sum[i]/nCells[i];
              fprintf( fp, "%16.15E  %i\n", avg, nCells[i] );
            }
            else if( weightType == MASS ){
              double avg = sum[i]/weight[i];
              fprintf( fp, "%16.15E  %16.15E\n", avg, weight[i] );
            } 
            else{   // weightType == NONE
              fprintf( fp, "%16.15E\n", sum[i] );
            }
          }
        }
        ~planarVar_double(){}
    };


    //______________________________________________________________________
    //  Class that holds the planar quantities      VECTOR
    class planarVar_Vector: public planarVarBase{

      //___________________________________
      //  MPI user defined function for computing sum for Uintah::Vector
      static void plusEqualVector( Vector * in,
                                   Vector * inOut,
                                   int  * len,
                                   MPI_Datatype * type)
      {
        for (auto i=0; i< *len; i++){
          inOut[i].x( inOut[i].x() + in[i].x() );
          inOut[i].y( inOut[i].y() + in[i].y() );
          inOut[i].z( inOut[i].z() + in[i].z() );
        }
      }

      //__________________________________
      private:
        std::vector<Vector> sum;

      public:

        //__________________________________
        void reserve()
        {
          Vector zero(0.);
          CC_pos.resize( nPlanes, Point(0,0,0) );
          sum.resize(    nPlanes, Vector(0,0,0) );
          weight.resize( nPlanes, 0 );
          nCells.resize( nPlanes, 0 );
        }
        //__________________________________
        void getPlanarSum( std::vector<Vector> & me ) { me = sum; }
        void setPlanarSum( std::vector<Vector> & me ) { sum = me; }

        //__________________________________
        void zero_all_vars()
        {
          for(unsigned i=0; i<sum.size(); i++ ){
            CC_pos[i] = Point(0,0,0);
            sum[i]    = Vector(0,0,0);
            weight[i] = 0;
            nCells[i] = 0;
          }
        }

        //__________________________________
        void ReduceVar( const int rank )
        {
          MPI_Datatype  mpitype;
          Uintah::MPI::Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
          Uintah::MPI::Type_commit( &mpitype );

          MPI_Op vector_add;
          MPI_Op_create( (MPI_User_function*) plusEqualVector, 1, &vector_add );

          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &sum.front(), nPlanes, mpitype, vector_add, rootRank, MPI_COMM_WORLD );
          } else {
            Uintah::MPI::Reduce(  &sum.front(), 0,            nPlanes, mpitype, vector_add, rootRank, MPI_COMM_WORLD );
          }
          MPI_Op_free( &vector_add );
        }


        //__________________________________
        void printAverage( FILE* & fp,
                           const int levelIndex,
                           const double simTime )
        {
          fprintf( fp,"# Level: %i \n", levelIndex );
          fprintf( fp,"# Simulation time: %16.15E \n", simTime );
          fprintf( fp,"# Plane location (x,y,z)           Average\n" );
          fprintf( fp,"# ________CC_loc.x_______________CC_loc.y______________CC_loc.z_____________ave.x__________________ave.y______________ave.z" );
          if( weightType == NCELLS ){
            fprintf( fp,"_______________nCells\n" );
          }
          else if( weightType == MASS ){
            fprintf( fp,"_________________weight\n" );
          }

          // loop over each plane, compute the ave and write to file
          for ( unsigned i =0; i< sum.size(); i++ ){
          
            fprintf( fp, "%16.15E  %16.15E  %16.15E ", CC_pos[i].x(), CC_pos[i].y(), CC_pos[i].z() );
            
            if( weightType == NCELLS ){
              Vector avg = sum[i]/Vector( nCells[i] );
              fprintf( fp, "%16.15E %16.15E %16.15E %i\n", avg.x(), avg.y(), avg.z(), nCells[i] );
            }
            else if( weightType == MASS ){
              Vector avg = sum[i]/Vector( weight[i] );
              fprintf( fp, "%16.15E %16.15E %16.15E %16.15E\n", avg.x(), avg.y(), avg.z(), weight[i] );
            }
            else{   // weightType == NONE
              fprintf( fp, "%16.15E %16.15E %16.15E\n", sum[i].x(), sum[i].y(), sum[i].z() );
            }
          }
        }
        ~planarVar_Vector(){}
    };

    // For each level there's a vector of planeVarBases
    //
    std::vector<  std::vector< std::shared_ptr< planarVarBase > > > d_allLevels_planarVars;  
    
          

    //______________________________________________________________________
    //
    //


    IntVector findCellIndex(const int i,
                            const int j,
                            const int k);

    bool isItTime( DataWarehouse * old_dw);

    bool isRightLevel( const int myLevel,
                       const int L_indx,
                       const LevelP& level);

    void initialize(const ProcessorGroup *,
                    const PatchSubset    * patches,
                    const MaterialSubset *,
                    DataWarehouse        *,
                    DataWarehouse        * new_dw);

    void zeroPlanarVars(const ProcessorGroup * pg,
                     const PatchSubset    * patches,
                     const MaterialSubset *,
                     DataWarehouse        * old_dw,
                     DataWarehouse        * new_dw);

    void computePlanarSums(const ProcessorGroup * pg,
                           const PatchSubset    * patches,
                           const MaterialSubset *,
                           DataWarehouse        * old_dw,
                           DataWarehouse        * new_dw);

    void sumOverAllProcs(const ProcessorGroup * pg,
                         const PatchSubset    * patches,
                         const MaterialSubset *,
                         DataWarehouse        * old_dw,
                         DataWarehouse        * new_dw);

    void writeToFiles(const ProcessorGroup  * pg,
                      const PatchSubset     * patches,
                      const MaterialSubset  *,
                      DataWarehouse         *,
                      DataWarehouse         * new_dw);

    void createFile(std::string & filename,
                    FILE*       & fp,
                    std::string & levelIndex);

    int createDirectory( mode_t mode,
                         const std::string & rootPath,
                         std::string       & path );

    template <class Tvar, class Ttype>
    void planarSum_Q( DataWarehouse  * new_dw,
                      std::shared_ptr< planarVarBase > analyzeVars,
                      const Patch    * patch,
                      GridIterator    iter );

    template <class Tvar, class Ttype>
    void planarSum_weight( DataWarehouse * new_dw,
                           std::shared_ptr< planarVarBase > analyzeVar,
                           const Patch   * patch );


    void planeIterator( const GridIterator& patchIter,
                        IntVector & lo,
                        IntVector & hi );


    // general labels
    class planeAverageLabel {
    public:
      VarLabel* lastCompTimeLabel;
      VarLabel* fileVarsStructLabel;
      VarLabel* weightLabel;
    };

    planeAverageLabel* d_lb;

    //__________________________________
    // global constants always begin with "d_"
    double d_writeFreq;
    double d_startTime;
    double d_stopTime;

    const Material*  d_matl;
    MaterialSet*     d_matl_set;
    std::set<std::string> d_isDirCreated;
    MaterialSubset*  d_zero_matl;

    const int d_MAXLEVELS {5};               // HARDCODED
    
    // Flag: has this rank has executed this task on this level
    std::vector< std::vector< bool > > d_progressVar;
    enum taskNames { INITIALIZE=0, ZERO=1, COMPUTE=2, SUM=3, N_TASKS=4 };

    enum orientation { XY, XZ, YZ };        // plane orientation
    orientation d_planeOrientation;

  };
}

#endif
