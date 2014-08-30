#include <iostream>

//--- SpatialOps includes ---//
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/IntVec.h>
#include <spatialops/structured/MemoryWindow.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/stencil/StencilBuilder.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/Nebo.h>

//-- boost includes ---//
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace po = boost::program_options;

using namespace SpatialOps;

int print_length = 10;

template<typename Field>
void print(Field const & given) {
  typename Field::const_iterator ig = given.begin();

  for(int i = 0; i < print_length; i++) {
    std::cout << *ig << std::endl;
    ++ig;
  };

  std::cout << std::endl;
};

template<typename Field>
void print_all(Field const & given) {
  typename Field::const_iterator ig = given.begin();

  MemoryWindow window = given.window_with_ghost();
  for(int kk = 0; kk < window.glob_dim(2); kk++) {
      for(int jj = 0; jj < window.glob_dim(1); jj++) {
          for(int ii = 0; ii < window.glob_dim(0); ii++, ++ig) {
              std::cout << *ig << " ";
          }
          std::cout <<std::endl;
      }
      std::cout <<std::endl;
  };
};

#define INT_EQU(FIELD, FIRST, SECOND, PRINT)                            \
    {                                                                   \
        FIELD::const_iterator fi = FIRST.interior_begin();              \
        FIELD::const_iterator si = SECOND.interior_begin();             \
        FIELD::const_iterator se = SECOND.interior_end();               \
        bool flag = true;                                               \
        int count = 0;                                                  \
                                                                        \
        while( si != se ) {                                             \
            if( *fi != *si ) {                                          \
                flag = false;                                           \
                if( PRINT ) {                                           \
                    std::cout << count                                  \
                              << ": " << *fi                            \
                              << " " << *si << std::endl;               \
                };                                                      \
            };                                                          \
            ++fi;                                                       \
            ++si;                                                       \
            ++count;                                                    \
        };                                                              \
                                                                        \
        if( flag ) std::cout << "Success!";                             \
        else std::cout << "Error!";                                     \
        std::cout << std::endl;                                         \
    }


#define RUN_TEST(TEST,							\
		 TYPE)							\
  boost::posix_time::ptime start( boost::posix_time::microsec_clock::universal_time() ); \
  boost::posix_time::ptime end( boost::posix_time::microsec_clock::universal_time() ); \
  int ii = 0;								\
  start = boost::posix_time::microsec_clock::universal_time();		\
									\
  for(; ii < number_of_runs; ii++) {					\
    TEST;								\
  };									\
									\
  end = boost::posix_time::microsec_clock::universal_time();		\
  std::cout << TYPE;							\
  std::cout << " runs: ";						\
  std::cout << number_of_runs;						\
  std::cout << " result: ";						\
  std::cout << (end - start).total_microseconds()*1e-6;			\
  std::cout << std::endl;

template<typename FieldType>
inline void evaluate_serial_example(FieldType & result,
				    FieldType const & phi,
				    FieldType const & dCoef,
				    IntVec const npts,
				    double const Lx,
				    double const Ly,
				    double const Lz,
				    int number_of_runs) {

    SpatialOps::OperatorDatabase opDB;
    SpatialOps::build_stencils(npts[0],
                                           npts[1],
                                           npts[2],
                                           Lx,
                                           Ly,
                                           Lz,
                                           opDB);

    typename BasicOpTypes<FieldType>::GradX* const gradXOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradX>();
    typename BasicOpTypes<FieldType>::GradY* const gradYOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradY>();
    typename BasicOpTypes<FieldType>::GradZ* const gradZOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradZ>();
    //typename BasicOpTypes<FieldType>::InterpC2FX* const interpXOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FX>();
    //typename BasicOpTypes<FieldType>::InterpC2FY* const interpYOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FY>();
    //typename BasicOpTypes<FieldType>::InterpC2FZ* const interpZOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FZ>();
    typename BasicOpTypes<FieldType>::DivX* const divXOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivX>();
    typename BasicOpTypes<FieldType>::DivY* const divYOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivY>();
    typename BasicOpTypes<FieldType>::DivZ* const divZOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivZ>();

    const MemoryWindow& w = phi.window_with_ghost();
    const GhostData& g = phi.get_ghost_data();
    const BoundaryCellInfo& bc = phi.boundary_info();
    typename FaceTypes<FieldType>::XFace  tmpFaceX( w, bc, g, NULL );
    typename FaceTypes<FieldType>::XFace tmpFaceX2( w, bc, g, NULL );
    FieldType tmpX( w, bc, g, NULL );
    typename FaceTypes<FieldType>::YFace  tmpFaceY( w, bc, g, NULL );
    typename FaceTypes<FieldType>::YFace tmpFaceY2( w, bc, g, NULL );
    FieldType tmpY( w, bc, g, NULL );
    typename FaceTypes<FieldType>::ZFace  tmpFaceZ( w, bc, g, NULL );
    typename FaceTypes<FieldType>::ZFace tmpFaceZ2( w, bc, g, NULL );
    FieldType tmpZ( w, bc, g, NULL );

    RUN_TEST(// X - direction
	     gradXOp_->apply_to_field( phi,    tmpFaceX  );
	     divXOp_->apply_to_field( tmpFaceX, tmpX );

	     // Y - direction
	     gradYOp_->apply_to_field( tmpX,    tmpFaceY  );
	     divYOp_->apply_to_field( tmpFaceY, tmpY );

	     // Z - direction
	     gradZOp_->apply_to_field( tmpY,    tmpFaceZ  );
	     divZOp_->apply_to_field( tmpFaceZ, tmpZ );

	     result <<= - tmpZ,
	     "old");

};

template<typename FieldType>
inline void evaluate_chaining_example(FieldType & result,
				      FieldType const & phi,
				      FieldType const & dCoef,
				      IntVec const npts,
				      double const Lx,
				      double const Ly,
				      double const Lz,
				      int number_of_runs) {


    SpatialOps::OperatorDatabase opDB;
    SpatialOps::build_stencils(npts[0],
                                           npts[1],
                                           npts[2],
                                           Lx,
					   Ly,
					   Lz,
                                           opDB);

    typename BasicOpTypes<FieldType>::GradX neboGradX = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradX>();
    typename BasicOpTypes<FieldType>::GradY neboGradY = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradY>();
    typename BasicOpTypes<FieldType>::GradZ neboGradZ = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradZ>();
    typename BasicOpTypes<FieldType>::InterpC2FX neboInterpX = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FX>();
    typename BasicOpTypes<FieldType>::InterpC2FY neboInterpY = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FY>();
    typename BasicOpTypes<FieldType>::InterpC2FZ neboInterpZ = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FZ>();
    typename BasicOpTypes<FieldType>::DivX neboDivX = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivX>();
    typename BasicOpTypes<FieldType>::DivY neboDivY = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivY>();
    typename BasicOpTypes<FieldType>::DivZ neboDivZ = *opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivZ>();

    RUN_TEST(result <<= (- neboDivZ(neboGradZ(
                                              neboDivY(neboGradY(
                                                                 neboDivX(neboGradX(phi))))))),
             "new");

};

int main(int iarg, char* carg[]) {
    typedef SVolField Field;

    bool test;
    int nx, ny, nz;
    int number_of_runs;
    double Lx, Ly, Lz;
#ifdef ENABLE_THREADS
  int thread_count;
#endif

    // parse the command line options input describing the problem
    {
        po::options_description desc("Supported Options");
	desc.add_options()
	  ( "help", "print help message" )
	  ( "nx", po::value<int>(&nx)->default_value(10), "Grid in x" )
	  ( "ny", po::value<int>(&ny)->default_value(10), "Grid in y" )
	  ( "nz", po::value<int>(&nz)->default_value(10), "Grid in z" )
	  ( "Lx", po::value<double>(&Lx)->default_value(1.0),"Length in x")
	  ( "Ly", po::value<double>(&Ly)->default_value(1.0),"Length in y")
	  ( "Lz", po::value<double>(&Lz)->default_value(1.0),"Length in z")
          ( "check", po::value<bool>(&test)->default_value(true),"Compare results of old and new versions")
#ifdef ENABLE_THREADS
      ( "tc", po::value<int>(&thread_count)->default_value(NTHREADS), "Number of threads for Nebo")
#endif
	  ( "runs", po::value<int>(&number_of_runs)->default_value(1), "Number of iterations of each test");

	po::variables_map args;
	po::store( po::parse_command_line(iarg,carg,desc), args );
	po::notify(args);

	if (args.count("help")) {
	    std::cout << desc << "\n";
	    return 1;
	}

#ifdef ENABLE_THREADS
    set_hard_thread_count(thread_count);
#endif
    }

    const GhostData ghost(1);
    const BoundaryCellInfo bc = BoundaryCellInfo::build<Field>(false,false,false);
    const MemoryWindow window( get_window_with_ghost(IntVec(nx,ny,nz),ghost,bc) );

    Field  a( window, bc, ghost, NULL );
    Field  b( window, bc, ghost, NULL );
    Field cr( window, bc, ghost, NULL );
    Field sr( window, bc, ghost, NULL );

    Field::iterator ia = a.begin();
    Field::iterator ib = b.begin();
    for(size_t kk = 0; kk < window.glob_dim(2); kk++) {
        for(size_t jj = 0; jj < window.glob_dim(1); jj++) {
            for(size_t ii = 0; ii < window.glob_dim(0); ii++, ++ia, ++ib) {
	      *ia = ii + jj * 2 + kk * 4;
	      *ib = ii + jj * 3 + kk * 5;
            }
        }
    };

    evaluate_serial_example(sr,
			    a,
			    b,
			    IntVec(nx,ny,nz),
			    Lx,
			    Ly,
			    Lz,
			    number_of_runs);

    evaluate_chaining_example(cr,
			      a,
			      b,
			      IntVec(nx,ny,nz),
			      Lx,
			      Ly,
			      Lz,
			      number_of_runs);

    if(test) {
        INT_EQU(Field, cr, sr, false);
    };

    return 0;
};
