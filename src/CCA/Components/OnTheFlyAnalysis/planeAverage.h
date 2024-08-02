/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include <vector>
#include <memory>

namespace Uintah {


/*______________________________________________________________________

GENERAL INFORMATION

   planeAverage.h

   This computes the spatial average over plane(s) in the domain.

   Todd Harman
   Department of Mechanical Engineering
   University of Utah
______________________________________________________________________*/



//______________________________________________________________________

  class planeAverage : public AnalysisModule {
  public:

    planeAverage(const ProcessorGroup   * myworld,
                 const MaterialManagerP   materialManager,
                 const ProblemSpecP     & module_spec,
                 const bool               parse_ups_variables,
                 const bool               writeOutput,
                 const int                ID );

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

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                             const LevelP & level) {};

    enum weightingType { NCELLS, MASS, NONE };

    static MPI_Comm d_my_MPI_COMM_WORLD;


  //______________________________________________________________________
  //
  public:

    //__________________________________
    //  This is a wrapper to create a vector of objects of different types(planarVar)
    struct planarVarBase{

      //___________________________________
      //  MPI user defined function for computing max for Uintah::Point
      //  we need this so each plane location has a non-zero value
      static void MPI_maxPoint( Point * in,
                                Point * inOut,
                                int   * len,
                                MPI_Datatype * type)
      {

        for (auto i=0; i< *len; i++){
          inOut[i] = Uintah::Max( inOut[i], in[i] );
        }
      }

      public:
        VarLabel* label;
        int matl           =-9;
        int level          =-9;
        int nPlanes        =-9;            // number of avg planes
        int floorIndex     =-9;            // cell index of bottom plane.  Not every level starts at 0
        const int rootRank = 0;
        std::vector<Point>  CC_pos;        // cell center position
        std::vector<double> weight;        // weighting to compute ave
        std::vector<int>    nCells;        // number of cells per plane.  Each plane could differ with AMR
        weightingType       weightType;
        std::string         fileDesc;      // description of variable used in the file header

        TypeDescription::Type baseType;
        TypeDescription::Type subType;

        //__________________________________
        void set_floorIndex(const int in) { floorIndex = in; }
        int  get_floorIndex() { return floorIndex; }

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
        // broadcast the value
        void ReduceBcastWeight( const int rank )
        {

          const MPI_Comm com = planeAverage::d_my_MPI_COMM_WORLD;  // readability

          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &weight.front(), nPlanes, MPI_DOUBLE, MPI_SUM, rootRank, com );
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &nCells.front(), nPlanes, MPI_INT,    MPI_SUM, rootRank, com );
          } else {
            Uintah::MPI::Reduce(  &weight.front(), 0,            nPlanes, MPI_DOUBLE, MPI_SUM, rootRank, com );
            Uintah::MPI::Reduce(  &nCells.front(), 0,            nPlanes, MPI_INT,    MPI_SUM, rootRank, com );
          }

          // broadcast the weight to all ranks
          Uintah::MPI::Bcast( &nCells.front(), nPlanes, MPI_INT,    rootRank, com);
          Uintah::MPI::Bcast( &weight.front(), nPlanes, MPI_DOUBLE, rootRank, com);
        }

        //__________________________________
        void setCC_pos( std::vector<Point>  & pos,
                        const unsigned lo,
                        const unsigned hi)
        {
          for ( auto z = lo; z<hi; z++ ) {
            const int i = z - floorIndex;
            CC_pos[i] = pos[i];
          }
        }

        //__________________________________
        void ReduceCC_pos( const int rank )
        {

          Point *P = nullptr;
          const TypeDescription* td = fun_getTypeDescription( P );
          MPI_Datatype Point_type = td->getMPIType();

          MPI_Op max_point;
          MPI_Op_create( (MPI_User_function*) MPI_maxPoint, 1, &max_point );

          const MPI_Comm com = planeAverage::d_my_MPI_COMM_WORLD;  // readability

          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &CC_pos.front(), nPlanes, Point_type, max_point, rootRank, com );
          } else {
            Uintah::MPI::Reduce(  &CC_pos.front(), 0,            nPlanes, Point_type, max_point, rootRank, com );
          }
          MPI_Op_free( &max_point );
        }

        //__________________________________
        //  common file header
        void printHeader(  FILE* & fp,
                           const Level* level,
                           const double simTime,
                           const std::string fileDesc)
        {
          int L_index = level->getIndex();
          BBox b;
          level->getInteriorSpatialRange( b );
          const Point bmin = b.min();
          const Point bmax = b.max();

          fprintf( fp, "# Level: %i nCells per plane: %i\n", L_index, nCells[0] );
          fprintf( fp, "# Level spatial range:\n" );
          fprintf( fp, "# %15.14E  %15.14E  %15.14E \n", bmin.x(), bmin.y(), bmin.z() );
          fprintf( fp, "# %15.14E  %15.14E  %15.14E \n", bmax.x(), bmax.y(), bmax.z() );

          IntVector lo;
          IntVector hi;
          level->findInteriorCellIndexRange( lo, hi );

          fprintf( fp, "# Level interior CC index range:  " );
          fprintf( fp, "# [%i %i %i] [%i %i %i] \n", lo.x(), lo.y(), lo.z(), hi.x(), hi.y(), hi.z() );
          fprintf( fp, "# Simulation time: %16.15E \n", simTime );

          fprintf( fp,"# Plane location (x,y,z)           Average\n" );
          fprintf( fp,"# ________CC_loc.x__________CC_loc.y______________CC_loc.z______________%s", fileDesc.c_str() );

          if( weightType == NCELLS ){
            fprintf( fp,"______________nCells\n" );
          }
          else if( weightType == MASS ){
            fprintf( fp,"______________weight\n" );
          }
        }

        //__________________________________
        //   VIRTUAL FUNCTIONS

        virtual void reserve() = 0;

        virtual std::shared_ptr<planarVarBase> clone() const = 0;

        // virtual templated functions are not allowed in C++11
        // instantiate the various types
        virtual void getPlanarAve( std::vector<double>  & ave ){}
        virtual void getPlanarAve( std::vector<Vector>  & ave ){}
        virtual void getPlanarAve( std::vector<Matrix3> & ave ){}

        virtual  void getPlanarSum( std::vector<double> & sum ){}
        virtual  void getPlanarSum( std::vector<Vector> & sum ){}
        virtual  void getPlanarSum( std::vector<Matrix3>& sum ){}

        virtual  void setPlanarSum( std::vector<double> & sum ){}
        virtual  void setPlanarSum( std::vector<Vector> & sum ){}
        virtual  void setPlanarSum( std::vector<Matrix3> & sum ){}

        virtual  void zero_all_vars(){}

        virtual  void ReduceBcastVar(const int rank ) = 0;

        virtual  void printAverage( FILE* & fp,
                                    const Level* level,
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
        std::vector<double> ave;

      public:

        planarVar_double(){
          fileDesc = "ave";
        }

        //__________________________________
        // this makes a deep copy
        virtual std::shared_ptr<planarVarBase> clone() const {
          return std::make_shared<planarVar_double>(*this);
        }

        //__________________________________
        void reserve()
        {
          CC_pos.resize(nPlanes, Point(-DBL_MAX,-DBL_MAX,-DBL_MAX) );
          ave.resize(   nPlanes, 0.);
          sum.resize(   nPlanes, 0.);
          weight.resize(nPlanes, 0.);
          nCells.resize(nPlanes, 0 );
        }

        //__________________________________
        void getPlanarSum( std::vector<double> & me ) { me = sum; }
        void setPlanarSum( std::vector<double> & me ) { sum = me; }

        //__________________________________
        void getPlanarAve( std::vector<double> & ave )
        {
          ave.resize( nPlanes, -9 );

          for ( unsigned i =0; i< sum.size(); i++ ){
            if( weightType == NCELLS ){
              ave[i] = sum[i]/nCells[i];
            }
            else if( weightType == MASS ){
              ave[i] = sum[i]/weight[i];
            }
            else{   // weightType == NONE
              ave[i] = sum[i];
            }
          }
        }

        //__________________________________
        void zero_all_vars()
        {
          for(unsigned i=0; i<sum.size(); i++ ){
            CC_pos[i] = Point( -DBL_MAX,-DBL_MAX,-DBL_MAX );
            ave[i]    = 0.0;
            sum[i]    = 0.0;
            weight[i] = 0.0;
            nCells[i] = 0;
          }
        }

        //__________________________________
        // Reduce the sum over all ranks and
        // broadcast the value.
        void ReduceBcastVar( const int rank )
        {
          const MPI_Comm com = planeAverage::d_my_MPI_COMM_WORLD;  // readability
          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &sum.front(), nPlanes, MPI_DOUBLE, MPI_SUM, rootRank, com );
          } else {
            Uintah::MPI::Reduce(  &sum.front(), 0,            nPlanes, MPI_DOUBLE, MPI_SUM, rootRank, com );
          }

          // broadcast the sum to all ranks
          Uintah::MPI::Bcast( &sum.front(), nPlanes, MPI_DOUBLE, rootRank, com);
        }

        //__________________________________
        void printAverage( FILE* & fp,
                           const Level* level,
                           const double simTime )
        {
          printHeader(fp, level, simTime, fileDesc);

          // loop over each plane, compute the ave and write to file
          for ( unsigned i =0; i< sum.size(); i++ ){

            fprintf( fp, "%15.14E  %15.14E  %15.14E ", CC_pos[i].x(), CC_pos[i].y(), CC_pos[i].z() );

            if( weightType == NCELLS ){
              double avg = sum[i]/nCells[i];
              fprintf( fp, "%15.14E  %i\n", avg, nCells[i] );
            }
            else if( weightType == MASS ){
              double avg = sum[i]/weight[i];
              fprintf( fp, "%15.14E  %15.14E\n", avg, weight[i] );
            }
            else{   // weightType == NONE
              fprintf( fp, "%15.14E\n", sum[i] );
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
        std::vector<Vector> ave;

      public:

        planarVar_Vector(){
          fileDesc = "ave.x__________________ave.y______________ave.z";
        }

        //__________________________________
        // this makes a deep copy
        virtual std::shared_ptr<planarVarBase> clone() const {
          return std::make_shared<planarVar_Vector>(*this);
        }

        //__________________________________
        void reserve()
        {
          Vector zero(0.);
          CC_pos.resize( nPlanes, Point(-DBL_MAX,-DBL_MAX,-DBL_MAX) );
          ave.resize(    nPlanes, Vector(0,0,0) );
          sum.resize(    nPlanes, Vector(0,0,0) );
          weight.resize( nPlanes, 0 );
          nCells.resize( nPlanes, 0 );
        }
        //__________________________________
        void getPlanarSum( std::vector<Vector> & me ) { me = sum; }
        void setPlanarSum( std::vector<Vector> & me ) { sum = me; }

        //__________________________________
        void getPlanarAve( std::vector<Vector> & ave )
        {
          ave.resize( nPlanes, Vector(-9) );

          for ( unsigned i =0; i< sum.size(); i++ ){
            if( weightType == NCELLS ){
              ave[i] = sum[i]/Vector( nCells[i] );
            }
            else if( weightType == MASS ){
              ave[i] = sum[i]/Vector( weight[i] );
            }
            else{   // weightType == NONE
              ave[i] = sum[i];
            }
          }
        }

        //__________________________________
        void zero_all_vars()
        {
          for(unsigned i=0; i<sum.size(); i++ ){
            CC_pos[i] = Point( -DBL_MAX,-DBL_MAX,-DBL_MAX );
            ave[i]    = Vector(0,0,0);
            sum[i]    = Vector(0,0,0);
            weight[i] = 0;
            nCells[i] = 0;
          }
        }

        //__________________________________
        // Reduce the sum over all ranks and
        // broadcast the value.
        void ReduceBcastVar( const int rank )
        {
          Vector *V = nullptr;
          const TypeDescription* td = fun_getTypeDescription( V );
          MPI_Datatype Vector_type = td->getMPIType();

          MPI_Op vector_add;
          MPI_Op_create( (MPI_User_function*) plusEqualVector, 1, &vector_add );

          const MPI_Comm com = planeAverage::d_my_MPI_COMM_WORLD;  // readability

          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &sum.front(), nPlanes, Vector_type, vector_add, rootRank, com );
          } else {
            Uintah::MPI::Reduce(  &sum.front(), 0,            nPlanes, Vector_type, vector_add, rootRank, com );
          }
          MPI_Op_free( &vector_add );

          // broadcast the sum to all ranks
          Uintah::MPI::Bcast( &sum.front(), nPlanes, Vector_type, rootRank, com);
        }


        //__________________________________
        void printAverage( FILE* & fp,
                           const Level* level,
                           const double simTime )
        {
          printHeader(fp, level, simTime, fileDesc);

          // loop over each plane, compute the ave and write to file
          for ( unsigned i =0; i< sum.size(); i++ ){

            fprintf( fp, "%15.14E  %15.14E  %15.14E ", CC_pos[i].x(), CC_pos[i].y(), CC_pos[i].z() );

            if( weightType == NCELLS ){
              Vector avg = sum[i]/Vector( nCells[i] );
              fprintf( fp, "%15.14E %15.14E %15.14E %i\n", avg.x(), avg.y(), avg.z(), nCells[i] );
            }
            else if( weightType == MASS ){
              Vector avg = sum[i]/Vector( weight[i] );
              fprintf( fp, "%15.14E %15.14E %15.14E %15.14E\n", avg.x(), avg.y(), avg.z(), weight[i] );
            }
            else{   // weightType == NONE
              fprintf( fp, "%15.14E %15.14E %15.14E\n", sum[i].x(), sum[i].y(), sum[i].z() );
            }
          }
        }
        ~planarVar_Vector(){}
    };

    //______________________________________________________________________
    //  Class that holds the planar quantities      Matrix3
    class planarVar_Matrix3: public planarVarBase{

      //___________________________________
      //  MPI user defined function for computing sum for Uintah::Matrix3
      static void plusEqualMatrix3( Matrix3 * in,
                                    Matrix3 * inOut,
                                    int  * len,
                                    MPI_Datatype * type)
      {
        for (auto l=0; l< *len; l++){
          for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
              inOut[l](i,j) += in[l](i,j);
            }
          }
        }
      }

      //__________________________________
      private:
        std::vector<Matrix3> sum;
        std::vector<Matrix3> ave;

      public:

        planarVar_Matrix3()
        {
          fileDesc = "";

          for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
              char a[50];
              sprintf(a, "ave.%i%i_______________", i+1,j+1);

              fileDesc += a;
            }
          }
          fileDesc.erase( fileDesc.length()-15 );         // remove the last dashed lines
        }

        //__________________________________
        // this makes a deep copy
        virtual std::shared_ptr<planarVarBase> clone() const {
          return std::make_shared<planarVar_Matrix3>(*this);
        }

        //__________________________________
        void reserve()
        {
          Matrix3 zero(0.);
          CC_pos.resize( nPlanes, Point(-DBL_MAX,-DBL_MAX,-DBL_MAX) );
          ave.resize(    nPlanes, Matrix3(0.0) );
          sum.resize(    nPlanes, Matrix3(0.0) );
          weight.resize( nPlanes, 0 );
          nCells.resize( nPlanes, 0 );
        }
        //__________________________________
        void getPlanarSum( std::vector<Matrix3> & me ) { me = sum; }
        void setPlanarSum( std::vector<Matrix3> & me ) { sum = me; }

        //__________________________________
        void getPlanarAve( std::vector<Matrix3> & ave )
        {
          ave.resize( nPlanes, Matrix3(-9) );

          for ( unsigned i =0; i< sum.size(); i++ ){
            if( weightType == NCELLS ){
              ave[i] = sum[i]/nCells[i];
            }
            else if( weightType == MASS ){
              ave[i] = sum[i]/weight[i];
            }
            else{   // weightType == NONE
              ave[i] = sum[i];
            }
          }
        }

        //__________________________________
        void zero_all_vars()
        {
          for(unsigned i=0; i<sum.size(); i++ ){
            CC_pos[i] = Point( -DBL_MAX,-DBL_MAX,-DBL_MAX );
            ave[i]    = Matrix3(0.0);
            sum[i]    = Matrix3(0.0);
            weight[i] = 0;
            nCells[i] = 0;
          }
        }

        //__________________________________
        // Reduce the sum over all ranks and
        // broadcast the value.
        void ReduceBcastVar( const int rank )
        {
          Matrix3 *m = nullptr;
          const TypeDescription* td = fun_getTypeDescription( m );
          MPI_Datatype  Matrix3_type = td->getMPIType();

          MPI_Op matrix3_add;
          MPI_Op_create( (MPI_User_function*) plusEqualMatrix3, 1, &matrix3_add );

          const MPI_Comm com = planeAverage::d_my_MPI_COMM_WORLD;  // readability

          if( rank == rootRank ){
            Uintah::MPI::Reduce(  MPI_IN_PLACE, &sum.front(), nPlanes, Matrix3_type, matrix3_add, rootRank, com );
          } else {
            Uintah::MPI::Reduce(  &sum.front(), 0,            nPlanes, Matrix3_type, matrix3_add, rootRank, com );
          }
          MPI_Op_free( &matrix3_add );

          // broadcast the sum to all ranks
          Uintah::MPI::Bcast( &sum.front(), nPlanes, Matrix3_type, rootRank, com);
        }

        //__________________________________
        //    print to file pointer ave
        void fprintAve( FILE* & fp, const Matrix3 avg)
        {
          for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
              fprintf( fp, "%15.14E ", avg(i,j));
            }
          }
        }

        //__________________________________
        void printAverage( FILE* & fp,
                           const Level* level,
                           const double simTime )
        {
          printHeader(fp, level, simTime, fileDesc);

          // loop over each plane, compute the ave and write to file
          for ( unsigned p =0; p< sum.size(); p++ ){

            fprintf( fp, "%15.14E  %15.14E  %15.14E ", CC_pos[p].x(), CC_pos[p].y(), CC_pos[p].z() );

            if( weightType == NCELLS ){
              Matrix3 avg = sum[p]/nCells[p];
              fprintAve( fp, avg );
              fprintf( fp, " %i\n", nCells[p] );
            }

            else if( weightType == MASS ){
              Matrix3 avg = sum[p]/weight[p];
              fprintAve( fp, avg );
              fprintf( fp, "%15.14E\n", weight[p] );
            }

            else{   // weightType == NONE
              fprintAve( fp, sum[p] );
            }
          }
        }
        ~planarVar_Matrix3(){}
    };

    // For each level there's a vector of planeVarBases
    //
    std::vector<  std::vector< std::shared_ptr< planarVarBase > > > d_allLevels_planarVars;

    //______________________________________________________________________
    //          public TASKS AND METHODS
  public:
    void sched_computePlanarAve( SchedulerP   & sched,
                                 const LevelP & level );

    void sched_writeToFiles( SchedulerP   &    sched,
                             const LevelP &    level,
                             const std::string  dirName );

    void sched_resetProgressVar( SchedulerP   & sched,
                                 const LevelP & level );

    void createMPICommunicator(const PatchSet* perProcPatches);


    template< class T >
    void getPlanarAve( const int        L_indx,
                       VarLabel *       label,
                       std::vector<T> & ave )
    {
      std::vector< std::shared_ptr< planarVarBase > > planarVars = d_allLevels_planarVars[L_indx];

      // find the correct label and return it's planar average
      for (unsigned int i =0 ; i < planarVars.size(); i++) {

        VarLabel* pv_label = planarVars[i]->label;
        if( label == pv_label ){
          planarVars[i]->getPlanarAve( ave );
          break;
        }
      }
    }


    IntVector transformCellIndex(const int i,
                                 const int j,
                                 const int k);

    void planeIterator( const GridIterator& patchIter,
                        IntVector & lo,
                        IntVector & hi );

    void setAllPlanes(const Level * level,
                      std::vector< std::shared_ptr< planarVarBase > > pv );

    void setAllLevels_planarVars( const int L_indx,
                                  std::vector< std::shared_ptr< planarVarBase > > pv )
    {
      d_allLevels_planarVars.at(L_indx) = pv;
    }


    //__________________________________
    //     PUBLIC:  VARIABLES
    MaterialSet*    d_matl_set;

    enum orientation { XY, XZ, YZ };        // plane orientation
    orientation d_planeOrientation;

        // general labels
    class planeAverageLabel {
    public:
      std::string lastCompTimeName;
      std::string fileVarsStructName;
      VarLabel* lastCompTimeLabel   {nullptr};
      VarLabel* fileVarsStructLabel {nullptr};
      VarLabel* weightLabel         {nullptr};
    };

    planeAverageLabel* d_lb;

  //______________________________________________________________________
  //
  private:
    void initialize(const ProcessorGroup *,
                    const PatchSubset    * patches,
                    const MaterialSubset *,
                    DataWarehouse        *,
                    DataWarehouse        * new_dw);


    void sched_initializePlanarVars( SchedulerP   & sched,
                                     const LevelP & level );

    void initializePlanarVars(const ProcessorGroup * pg,
                     const PatchSubset    * patches,
                     const MaterialSubset *,
                     DataWarehouse        * old_dw,
                     DataWarehouse        * new_dw);


    void sched_computePlanarSums( SchedulerP   & sched,
                                  const LevelP & level );

    void computePlanarSums(const ProcessorGroup * pg,
                           const PatchSubset    * patches,
                           const MaterialSubset *,
                           DataWarehouse        * old_dw,
                           DataWarehouse        * new_dw);


    void sched_sumOverAllProcs( SchedulerP   & sched,
                                const LevelP & level);

    void sumOverAllProcs(const ProcessorGroup * pg,
                         const PatchSubset    * patches,
                         const MaterialSubset *,
                         DataWarehouse        * old_dw,
                         DataWarehouse        * new_dw);


    void writeToFiles(const ProcessorGroup  * pg,
                      const PatchSubset     * patches,
                      const MaterialSubset  *,
                      DataWarehouse         *,
                      DataWarehouse         * new_dw,
                      const std::string       dirName);

    void updateTimeVar( const ProcessorGroup* pg,
                        const PatchSubset   * patches,
                        const MaterialSubset* matSubSet,
                        DataWarehouse       * old_dw,
                        DataWarehouse       * new_dw );

    template <class Tvar, class Ttype>
    void planarSum_Q( DataWarehouse  * new_dw,
                      std::shared_ptr< planarVarBase > analyzeVars,
                      const Patch    * patch,
                      GridIterator    iter );

    template <class Tvar, class Ttype>
    void planarSum_weight( DataWarehouse * new_dw,
                           std::shared_ptr< planarVarBase > analyzeVar,
                           const Patch   * patch );

    void resetProgressVar(const ProcessorGroup * ,
                          const PatchSubset    * patches,
                          const MaterialSubset *,
                          DataWarehouse        *,
                          DataWarehouse        *);

    void createFile(const std::string & filename,
                    const std::string & levelIndex,
                    FILE*       & fp );


    //__________________________________
    // global constants always begin with "d_"
    std::string d_className;                   // identifier for each instantiation of this class
    bool   d_parse_ups_variables;              // parse ups file to define d_allLevels_planarVars
                                               // this switch is needed for meanTurbFluxes module
    bool   d_writeOutput;

    const Material*  d_matl;

    const int d_MAXLEVELS {5};               // HARDCODED

    // Flag: has this rank has executed this task on this level
    std::vector< std::vector< bool > > d_progressVar;
    enum taskNames { INITIALIZE=0, ZERO=1, SUM=2, N_TASKS=3 };

    MaterialSubset* d_matl_subSet;

  };
}

#endif
