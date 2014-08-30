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

#define build_stencil_point(f, stencil_offset)                                  \
    (FieldType(MemoryWindow(f.window_without_ghost().glob_dim(),                \
                            f.window_without_ghost().offset() + stencil_offset, \
                            f.window_without_ghost().extent()),                 \
	       f) )

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
    typename BasicOpTypes<FieldType>::InterpC2FX* const interpXOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FX>();
    typename BasicOpTypes<FieldType>::InterpC2FY* const interpYOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FY>();
    typename BasicOpTypes<FieldType>::InterpC2FZ* const interpZOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FZ>();
    typename BasicOpTypes<FieldType>::DivX* const divXOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivX>();
    typename BasicOpTypes<FieldType>::DivY* const divYOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivY>();
    typename BasicOpTypes<FieldType>::DivZ* const divZOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivZ>();

    MemoryWindow const w = phi.window_with_ghost();
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
	     gradXOp_  ->apply_to_field( phi,    tmpFaceX  );
	     interpXOp_->apply_to_field( dCoef, tmpFaceX2 );
	     tmpFaceX <<= tmpFaceX * tmpFaceX2;
	     divXOp_->apply_to_field( tmpFaceX, tmpX );

	     // Y - direction
	     gradYOp_  ->apply_to_field( phi,    tmpFaceY  );
	     interpYOp_->apply_to_field( dCoef, tmpFaceY2 );
	     tmpFaceY <<= tmpFaceY * tmpFaceY2;
	     divYOp_->apply_to_field( tmpFaceY, tmpY );

	     // Z - direction
	     gradZOp_  ->apply_to_field( phi,    tmpFaceZ  );
	     interpZOp_->apply_to_field( dCoef, tmpFaceZ2 );
	     tmpFaceZ <<= tmpFaceZ * tmpFaceZ2;
	     divZOp_->apply_to_field( tmpFaceZ, tmpZ );

	     result <<= - tmpX - tmpY - tmpZ,
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

    typename BasicOpTypes<FieldType>::GradX* const gradXOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradX>();
    typename BasicOpTypes<FieldType>::GradY* const gradYOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradY>();
    typename BasicOpTypes<FieldType>::GradZ* const gradZOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::GradZ>();
    typename BasicOpTypes<FieldType>::InterpC2FX* const interpXOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FX>();
    typename BasicOpTypes<FieldType>::InterpC2FY* const interpYOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FY>();
    typename BasicOpTypes<FieldType>::InterpC2FZ* const interpZOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::InterpC2FZ>();
    typename BasicOpTypes<FieldType>::DivX* const divXOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivX>();
    typename BasicOpTypes<FieldType>::DivY* const divYOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivY>();
    typename BasicOpTypes<FieldType>::DivZ* const divZOp_ = opDB.retrieve_operator<typename BasicOpTypes<FieldType>::DivZ>();

    IntVec const neutral = IntVec(0,0,0);
    IntVec const neg_X = IntVec(-1,0,0);
    IntVec const pos_X = IntVec(1,0,0);
    IntVec const neg_Y = IntVec(0,-1,0);
    IntVec const pos_Y = IntVec(0,1,0);
    IntVec const neg_Z = IntVec(0,0,-1);
    IntVec const pos_Z = IntVec(0,0,1);

    RUN_TEST(FieldType r = build_stencil_point(result, neutral);

	     FieldType const phi_xlxl = build_stencil_point(phi, neg_X);
	     FieldType const phi_xlxh = build_stencil_point(phi, neutral);
	     FieldType const phi_xhxl = build_stencil_point(phi, neutral);
	     FieldType const phi_xhxh = build_stencil_point(phi, pos_X);
	     FieldType const phi_ylyl = build_stencil_point(phi, neg_Y);
	     FieldType const phi_ylyh = build_stencil_point(phi, neutral);
	     FieldType const phi_yhyl = build_stencil_point(phi, neutral);
	     FieldType const phi_yhyh = build_stencil_point(phi, pos_Y);
	     FieldType const phi_zlzl = build_stencil_point(phi, neg_Z);
	     FieldType const phi_zlzh = build_stencil_point(phi, neutral);
	     FieldType const phi_zhzl = build_stencil_point(phi, neutral);
	     FieldType const phi_zhzh = build_stencil_point(phi, pos_Z);

	     FieldType const dCoef_xlxl = build_stencil_point(dCoef, neg_X);
	     FieldType const dCoef_xlxh = build_stencil_point(dCoef, neutral);
	     FieldType const dCoef_xhxl = build_stencil_point(dCoef, neutral);
	     FieldType const dCoef_xhxh = build_stencil_point(dCoef, pos_X);
	     FieldType const dCoef_ylyl = build_stencil_point(dCoef, neg_Y);
	     FieldType const dCoef_ylyh = build_stencil_point(dCoef, neutral);
	     FieldType const dCoef_yhyl = build_stencil_point(dCoef, neutral);
	     FieldType const dCoef_yhyh = build_stencil_point(dCoef, pos_Y);
	     FieldType const dCoef_zlzl = build_stencil_point(dCoef, neg_Z);
	     FieldType const dCoef_zlzh = build_stencil_point(dCoef, neutral);
	     FieldType const dCoef_zhzl = build_stencil_point(dCoef, neutral);
	     FieldType const dCoef_zhzh = build_stencil_point(dCoef, pos_Z);

	     double gXcl = gradXOp_->get_minus_coef();
	     double gXch = gradXOp_->get_plus_coef();
	     double iXcl = interpXOp_->get_minus_coef();
	     double iXch = interpXOp_->get_plus_coef();
	     double dXcl = divXOp_->get_minus_coef();
	     double dXch = divXOp_->get_plus_coef();
	     double gYcl = gradYOp_->get_minus_coef();
	     double gYch = gradYOp_->get_plus_coef();
	     double iYcl = interpYOp_->get_minus_coef();
	     double iYch = interpYOp_->get_plus_coef();
	     double dYcl = divYOp_->get_minus_coef();
	     double dYch = divYOp_->get_plus_coef();
	     double gZcl = gradZOp_->get_minus_coef();
	     double gZch = gradZOp_->get_plus_coef();
	     double iZcl = interpZOp_->get_minus_coef();
	     double iZch = interpZOp_->get_plus_coef();
	     double dZcl = divZOp_->get_minus_coef();
	     double dZch = divZOp_->get_plus_coef();

	     r <<= (- (dXcl * ((gXcl * phi_xlxl + gXch * phi_xhxl) * (iXcl * dCoef_xlxl + iXch * dCoef_xhxl)) +
		       dXch * ((gXcl * phi_xlxh + gXch * phi_xhxh) * (iXcl * dCoef_xlxh + iXch * dCoef_xhxh)))
		    - (dYcl * ((gYcl * phi_ylyl + gYch * phi_yhyl) * (iYcl * dCoef_ylyl + iYch * dCoef_yhyl)) +
		       dYch * ((gYcl * phi_ylyh + gYch * phi_yhyh) * (iYcl * dCoef_ylyh + iYch * dCoef_yhyh)))
		    - (dZcl * ((gZcl * phi_zlzl + gZch * phi_zhzl) * (iZcl * dCoef_zlzl + iZch * dCoef_zhzl)) +
		       dZch * ((gZcl * phi_zlzh + gZch * phi_zhzh) * (iZcl * dCoef_zlzh + iZch * dCoef_zhzh)))),
	     "new");

};

int main(int iarg, char* carg[]) {
    typedef SVolField Field;

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
    return 0;
};
