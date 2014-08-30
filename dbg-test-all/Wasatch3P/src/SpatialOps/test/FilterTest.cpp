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

    const int nghost = 1;
    const GhostData ghost(nghost);
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

  if( dim[0]>1 && dim[1]>1 && dim[2]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter3DStencilCollection::StPtCollection, SVolField, SVolField> filter;
      RUN_TEST(filter.apply_to_field( a, cr ), "filter");
  } else if( dim[0]>1 && dim[1]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter2DXYStencilCollection::StPtCollection, SVolField, SVolField> filter;
      RUN_TEST(filter.apply_to_field( a, cr ), "filter");
  } else if( dim[0]>1 && dim[2]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter2DXZStencilCollection::StPtCollection, SVolField, SVolField> filter;
      RUN_TEST(filter.apply_to_field( a, cr ), "filter");
  } else if( dim[1]>1 && dim[2]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter2DYZStencilCollection::StPtCollection, SVolField, SVolField> filter;
      RUN_TEST(filter.apply_to_field( a, cr ), "filter");
  } else if( dim[0]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter1DXStencilCollection::StPtCollection, SVolField, SVolField> filter;
      RUN_TEST(filter.apply_to_field( a, cr ), "filter");
  } else if( dim[1]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter1DYStencilCollection::StPtCollection, SVolField, SVolField> filter;
      RUN_TEST(filter.apply_to_field( a, cr ), "filter");
  } else if( dim[2]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter1DZStencilCollection::StPtCollection, SVolField, SVolField> filter;
      RUN_TEST(filter.apply_to_field( a, cr ), "filter");
  }

    return 0;
};
