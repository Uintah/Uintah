#include <iostream>

//--- SpatialOps includes ---//
#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/Nebo.h>

//-- boost includes ---//
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace po = boost::program_options;

using namespace SpatialOps;

int main( int iarg, char* carg[] )
{
  typedef SpatialOps::SVolField Field;

  std::vector<int> npts(3,1);
  int number_of_runs;
#ifdef ENABLE_THREADS
  int thread_count;
#endif

  // parse the command line options input describing the problem
  {
    po::options_description desc("Supported Options");
    desc.add_options()
      ( "help", "print help message" )
      ( "nx", po::value<int>(&npts[0])->default_value(10), "Grid in x" )
      ( "ny", po::value<int>(&npts[1])->default_value(10), "Grid in y" )
      ( "nz", po::value<int>(&npts[2])->default_value(10), "Grid in z" )
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
  const BoundaryCellInfo bc = BoundaryCellInfo::build<Field>(true,true,true);
  const MemoryWindow window( SpatialOps::get_window_with_ghost(npts,ghost,bc) );

  // build fields
  Field f01  ( window, bc, ghost, NULL );
  Field f01_ ( window, bc, ghost, NULL );
  Field f01__( window, bc, ghost, NULL );
  Field f01_3( window, bc, ghost, NULL );
  Field f01_4( window, bc, ghost, NULL );
  Field f01_5( window, bc, ghost, NULL );
  Field f01_6( window, bc, ghost, NULL );
  Field f02  ( window, bc, ghost, NULL );
  Field f02_ ( window, bc, ghost, NULL );
  Field f03  ( window, bc, ghost, NULL );
  Field f03_ ( window, bc, ghost, NULL );
  Field f04  ( window, bc, ghost, NULL );
  Field f04_ ( window, bc, ghost, NULL );
  Field f04__( window, bc, ghost, NULL );
  Field f04_3( window, bc, ghost, NULL );
  Field f05  ( window, bc, ghost, NULL );
  Field f05_ ( window, bc, ghost, NULL );
  Field f06  ( window, bc, ghost, NULL );
  Field f06_ ( window, bc, ghost, NULL );
  Field f07  ( window, bc, ghost, NULL );
  Field f07_ ( window, bc, ghost, NULL );
  Field f07__( window, bc, ghost, NULL );
  Field f07_3( window, bc, ghost, NULL );
  Field f08  ( window, bc, ghost, NULL );
  Field f08_ ( window, bc, ghost, NULL );
  Field f09  ( window, bc, ghost, NULL );
  Field f09_ ( window, bc, ghost, NULL );
  Field f10  ( window, bc, ghost, NULL );
  Field f10_ ( window, bc, ghost, NULL );
  Field f10__( window, bc, ghost, NULL );
  Field f10_3( window, bc, ghost, NULL );
  Field f10_4( window, bc, ghost, NULL );
  Field f10_5( window, bc, ghost, NULL );
  Field f11  ( window, bc, ghost, NULL );
  Field f11_ ( window, bc, ghost, NULL );
  Field f12  ( window, bc, ghost, NULL );
  Field f12_ ( window, bc, ghost, NULL );
  Field f13  ( window, bc, ghost, NULL );
  Field f13_ ( window, bc, ghost, NULL );
  Field f13__( window, bc, ghost, NULL );
  Field f13_3( window, bc, ghost, NULL );
  Field f14  ( window, bc, ghost, NULL );
  Field f14_ ( window, bc, ghost, NULL );
  Field f15  ( window, bc, ghost, NULL );
  Field f15_ ( window, bc, ghost, NULL );
  Field f16  ( window, bc, ghost, NULL );
  Field f16_ ( window, bc, ghost, NULL );
  Field f16__( window, bc, ghost, NULL );
  Field f16_3( window, bc, ghost, NULL );
  Field f17  ( window, bc, ghost, NULL );
  Field f17_ ( window, bc, ghost, NULL );
  Field f18  ( window, bc, ghost, NULL );
  Field f18_ ( window, bc, ghost, NULL );
  Field f19  ( window, bc, ghost, NULL );
  Field f19_ ( window, bc, ghost, NULL );
  Field f19__( window, bc, ghost, NULL );
  Field f19_3( window, bc, ghost, NULL );
  Field f19_4( window, bc, ghost, NULL );
  Field f19_5( window, bc, ghost, NULL );
  Field f19_6( window, bc, ghost, NULL );
  Field f20  ( window, bc, ghost, NULL );
  Field f20_ ( window, bc, ghost, NULL );
  Field f21  ( window, bc, ghost, NULL );
  Field f21_ ( window, bc, ghost, NULL );
  Field f22  ( window, bc, ghost, NULL );
  Field f22_ ( window, bc, ghost, NULL );
  Field f22__( window, bc, ghost, NULL );
  Field f22_3( window, bc, ghost, NULL );
  Field f23  ( window, bc, ghost, NULL );
  Field f23_ ( window, bc, ghost, NULL );
  Field f24  ( window, bc, ghost, NULL );
  Field f24_ ( window, bc, ghost, NULL );
  Field f25  ( window, bc, ghost, NULL );
  Field f25_ ( window, bc, ghost, NULL );
  Field f25__( window, bc, ghost, NULL );
  Field f25_3( window, bc, ghost, NULL );
  Field f26  ( window, bc, ghost, NULL );
  Field f26_ ( window, bc, ghost, NULL );
  Field f27  ( window, bc, ghost, NULL );
  Field f27_ ( window, bc, ghost, NULL );
  Field f28  ( window, bc, ghost, NULL );
  Field f28_ ( window, bc, ghost, NULL );
  Field f28__( window, bc, ghost, NULL );
  Field f28_3( window, bc, ghost, NULL );
  Field f28_4( window, bc, ghost, NULL );
  Field f28_5( window, bc, ghost, NULL );
  Field f29  ( window, bc, ghost, NULL );
  Field f29_ ( window, bc, ghost, NULL );
  Field f30  ( window, bc, ghost, NULL );
  Field f30_ ( window, bc, ghost, NULL );
  Field f31  ( window, bc, ghost, NULL );
  Field f31_ ( window, bc, ghost, NULL );
  Field f31__( window, bc, ghost, NULL );
  Field f31_3( window, bc, ghost, NULL );
  Field f32  ( window, bc, ghost, NULL );
  Field f32_ ( window, bc, ghost, NULL );
  Field f33  ( window, bc, ghost, NULL );
  Field f33_ ( window, bc, ghost, NULL );
  Field f34  ( window, bc, ghost, NULL );
  Field f34_ ( window, bc, ghost, NULL );
  Field f34__( window, bc, ghost, NULL );
  Field f34_3( window, bc, ghost, NULL );
  Field f35  ( window, bc, ghost, NULL );
  Field f35_ ( window, bc, ghost, NULL );
  Field f36  ( window, bc, ghost, NULL );
  Field f36_ ( window, bc, ghost, NULL );
  Field result(window, bc, ghost, NULL );
  
  Field::iterator if01 = f01.begin();
  Field::iterator if02 = f02.begin();
  Field::iterator if03 = f03.begin();
  Field::iterator if04 = f04.begin();
  Field::iterator if05 = f05.begin();
  Field::iterator if06 = f06.begin();
  Field::iterator if07 = f07.begin();
  Field::iterator if08 = f08.begin();
  Field::iterator if09 = f09.begin();
  Field::iterator if10 = f10.begin();
  Field::iterator if11 = f11.begin();
  Field::iterator if12 = f12.begin();
  Field::iterator if13 = f13.begin();
  Field::iterator if14 = f14.begin();
  Field::iterator if15 = f15.begin();
  Field::iterator if16 = f16.begin();
  Field::iterator if17 = f17.begin();
  Field::iterator if18 = f18.begin();
  Field::iterator if19 = f19.begin();
  Field::iterator if20 = f20.begin();
  Field::iterator if21 = f21.begin();
  Field::iterator if22 = f22.begin();
  Field::iterator if23 = f23.begin();
  Field::iterator if24 = f24.begin();
  Field::iterator if25 = f25.begin();
  Field::iterator if26 = f26.begin();
  Field::iterator if27 = f27.begin();
  Field::iterator if28 = f28.begin();
  Field::iterator if29 = f29.begin();
  Field::iterator if30 = f30.begin();
  Field::iterator if31 = f31.begin();
  Field::iterator if32 = f32.begin();
  Field::iterator if33 = f33.begin();
  Field::iterator if34 = f34.begin();
  Field::iterator if35 = f35.begin();
  Field::iterator if36 = f36.begin();

  int size = npts[0] * npts[1] * npts[2];
  for(int ii = 0;
      if01!=f01.end();
      ++ii, ++if01, ++if02, ++if03, ++if04, ++if05, ++if06, ++if07, ++if08, ++if09,
	++if10, ++if11, ++if12, ++if13, ++if14, ++if15, ++if16, ++if17, ++if18, ++if19,
	++if20, ++if21, ++if22, ++if23, ++if24, ++if25, ++if26, ++if27, ++if28, ++if29,
          ++if30, ++if31, ++if32, ++if33, ++if34, ++if35, ++if36 ) {
    *if01 = ii + size * 1;
    *if02 = ii + size * 2;
    *if03 = ii + size * 3;
    *if04 = ii + size * 4;
    *if05 = ii + size * 5;
    *if06 = ii + size * 6;
    *if07 = ii + size * 7;
    *if08 = ii + size * 8;
    *if09 = ii + size * 9;
    *if10 = ii + size * 10;
    *if11 = ii + size * 11;
    *if12 = ii + size * 12;
    *if13 = ii + size * 13;
    *if14 = ii + size * 14;
    *if15 = ii + size * 15;
    *if16 = ii + size * 16;
    *if17 = ii + size * 17;
    *if18 = ii + size * 18;
    *if19 = ii + size * 19;
    *if20 = ii + size * 20;
    *if21 = ii + size * 21;
    *if22 = ii + size * 22;
    *if23 = ii + size * 23;
    *if24 = ii + size * 24;
    *if25 = ii + size * 25;
    *if26 = ii + size * 26;
    *if27 = ii + size * 27;
    *if28 = ii + size * 28;
    *if29 = ii + size * 29;
    *if30 = ii + size * 30;
    *if31 = ii + size * 31;
    *if32 = ii + size * 32;
    *if33 = ii + size * 33;
    *if34 = ii + size * 34;
    *if35 = ii + size * 35;
    *if36 = ii + size * 36;
  }

  boost::posix_time::ptime start( boost::posix_time::microsec_clock::universal_time() );
  boost::posix_time::ptime end( boost::posix_time::microsec_clock::universal_time() );
  int ii;

#define RUN_TESTS(TEST,							\
		  TYPE)							\
  ii = 0;								\
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


  //this is to warm up the system:
  result <<= ((((sin(f01) - sin(f02) - sin(f03)) + (sin(f04) - sin(f05) - sin(f06)) + (sin(f07) - sin(f08) - sin(f09))) *
	       ((sin(f10) - sin(f11) - sin(f12)) + (sin(f13) - sin(f14) - sin(f15)) + (sin(f16) - sin(f17) - sin(f18)))) /
	      (((sin(f19) - sin(f20) - sin(f21)) + (sin(f22) - sin(f23) - sin(f24)) + (sin(f25) - sin(f26) - sin(f27))) *
	       ((sin(f28) - sin(f29) - sin(f30)) + (sin(f31) - sin(f32) - sin(f33)) + (sin(f34) - sin(f35) - sin(f36)))));


  //1 Loop
  RUN_TESTS(result <<= ((((sin(f01) - sin(f02) - sin(f03)) + (sin(f04) - sin(f05) - sin(f06)) + (sin(f07) - sin(f08) - sin(f09))) *
			 ((sin(f10) - sin(f11) - sin(f12)) + (sin(f13) - sin(f14) - sin(f15)) + (sin(f16) - sin(f17) - sin(f18)))) /
			(((sin(f19) - sin(f20) - sin(f21)) + (sin(f22) - sin(f23) - sin(f24)) + (sin(f25) - sin(f26) - sin(f27))) *
			 ((sin(f28) - sin(f29) - sin(f30)) + (sin(f31) - sin(f32) - sin(f33)) + (sin(f34) - sin(f35) - sin(f36))))),
	    "1-loop");

  //3 Loop
  RUN_TESTS(f01_ <<= (((sin(f01) - sin(f02) - sin(f03)) + (sin(f04) - sin(f05) - sin(f06)) + (sin(f07) - sin(f08) - sin(f09))) *
		      ((sin(f10) - sin(f11) - sin(f12)) + (sin(f13) - sin(f14) - sin(f15)) + (sin(f16) - sin(f17) - sin(f18))));
	    f19_ <<= (((sin(f19) - sin(f20) - sin(f21)) + (sin(f22) - sin(f23) - sin(f24)) + (sin(f25) - sin(f26) - sin(f27))) *
		      ((sin(f28) - sin(f29) - sin(f30)) + (sin(f31) - sin(f32) - sin(f33)) + (sin(f34) - sin(f35) - sin(f36))));
	    result <<= f01_ / f19_,
	    "3-loop");

  //5 Loop
  RUN_TESTS(f01_ <<= ((sin(f01) - sin(f02) - sin(f03)) + (sin(f04) - sin(f05) - sin(f06)) + (sin(f07) - sin(f08) - sin(f09)));
	    f10_ <<= ((sin(f10) - sin(f11) - sin(f12)) + (sin(f13) - sin(f14) - sin(f15)) + (sin(f16) - sin(f17) - sin(f18)));
	    f19_ <<= ((sin(f19) - sin(f20) - sin(f21)) + (sin(f22) - sin(f23) - sin(f24)) + (sin(f25) - sin(f26) - sin(f27)));
	    f28_ <<= ((sin(f28) - sin(f29) - sin(f30)) + (sin(f31) - sin(f32) - sin(f33)) + (sin(f34) - sin(f35) - sin(f36)));
	    result <<= (f01_ * f10_) / (f19_ * f28_),
	    "5-loop");

  //13 Loop
  RUN_TESTS(f01_ <<= (sin(f01) - sin(f02) - sin(f03));
	    f04_ <<= (sin(f04) - sin(f05) - sin(f06));
	    f07_ <<= (sin(f07) - sin(f08) - sin(f09));
	    f10_ <<= (sin(f10) - sin(f11) - sin(f12));
	    f13_ <<= (sin(f13) - sin(f14) - sin(f15));
	    f16_ <<= (sin(f16) - sin(f17) - sin(f18));
	    f19_ <<= (sin(f19) - sin(f20) - sin(f21));
	    f22_ <<= (sin(f22) - sin(f23) - sin(f24));
	    f25_ <<= (sin(f25) - sin(f26) - sin(f27));
	    f28_ <<= (sin(f28) - sin(f29) - sin(f30));
	    f31_ <<= (sin(f31) - sin(f32) - sin(f33));
	    f34_ <<= (sin(f34) - sin(f35) - sin(f36));
	    result <<= ((f01_ + f04_ + f07_) * (f10_ + f13_ + f16_)) / ((f19_ + f22_ + f25_) * (f28_ + f31_ + f34_)),
	    "13-loop");

  //35 Loop
  RUN_TESTS(f01_ <<= (sin(f01) - sin(f02));
	    f01__ <<= (f01_ - sin(f03));
	    f04_ <<= (sin(f04) - sin(f05));
	    f04__ <<= (f04_ - sin(f06));
	    f07_ <<= (sin(f07) - sin(f08));
	    f07__ <<= (f07_ - sin(f09));
	    f10_ <<= (sin(f10) - sin(f11));
	    f10__ <<= (f10_ - sin(f12));
	    f13_ <<= (sin(f13) - sin(f14));
	    f13__ <<= (f13_ - sin(f15));
	    f16_ <<= (sin(f16) - sin(f17));
	    f16__ <<= (f16_ - sin(f18));
	    f19_ <<= (sin(f19) - sin(f20));
	    f19__ <<= (f19_ - sin(f21));
	    f22_ <<= (sin(f22) - sin(f23));
	    f22__ <<= (f22_ - sin(f24));
	    f25_ <<= (sin(f25) - sin(f26));
	    f25__ <<= (f25_ - sin(f27));
	    f28_ <<= (sin(f28) - sin(f29));
	    f28__ <<= (f28_ - sin(f30));
	    f31_ <<= (sin(f31) - sin(f32));
	    f31__ <<= (f31_ - sin(f33));
	    f34_ <<= (sin(f34) - sin(f35));
	    f34__ <<= (f34_ - sin(f36));
	    f04_3 <<= (f04__ + f07__);
	    f01_3 <<= (f01__ + f04_3);
	    f13_3 <<= (f13__ + f16__);
	    f10_3 <<= (f10__ + f13_3);
	    f22_3 <<= (f22__ + f25__);
	    f19_3 <<= (f19__ + f22_3);
	    f31_3 <<= (f31__ + f34__);
	    f28_3 <<= (f28__ + f31_3);
	    f01_4 <<= (f01_3 * f10_3);
	    f19_4 <<= (f19_3 * f28_3);
	    result <<= (f01_4 / f19_4),
	    "35-loop");

  //71 Loop
  RUN_TESTS(f01_ <<= sin(f01);
	    f02_ <<= sin(f02);
	    f01__ <<= (f01_ - f02_);
	    f03_ <<= sin(f03);
	    f01_3 <<= (f01__ - f03_);
	    f04_ <<= sin(f04);
	    f05_ <<= sin(f05);
	    f04__ <<= (f04_ - f05_);
	    f06_ <<= sin(f06);
	    f04_3 <<= (f04__ - f06_);
	    f01_4 <<= (f01_3 + f04_3);
	    f07_ <<= sin(f07);
	    f08_ <<= sin(f08);
	    f07__ <<= (f07_ - f08_);
	    f09_ <<= sin(f09);
	    f07_3 <<= (f07__ - f09_);
	    f01_5 <<= (f01_4 + f07_3);
	    f10_ <<= sin(f10);
	    f11_ <<= sin(f11);
	    f10__ <<= (f10_ - f11_);
	    f12_ <<= sin(f12);
	    f10_3 <<= (f10__ - f12_);
	    f13_ <<= sin(f13);
	    f14_ <<= sin(f14);
	    f13__ <<= (f13_ - f14_);
	    f15_ <<= sin(f15);
	    f13_3 <<= (f13__ - f15_);
	    f10_4 <<= (f10_3 + f13_3);
	    f16_ <<= sin(f16);
	    f17_ <<= sin(f17);
	    f16__ <<= (f16_ - f17_);
	    f18_ <<= sin(f18);
	    f16_3 <<= (f16__ - f18_);
	    f10_5 <<= (f10_4 + f16_3);
	    f01_6 <<= (f01_5 * f10_5);
	    f19_ <<= sin(f19);
	    f20_ <<= sin(f20);
	    f19__ <<= (f19_ - f20_);
	    f21_ <<= sin(f21);
	    f19_3 <<= (f19__ - f21_);
	    f22_ <<= sin(f22);
	    f23_ <<= sin(f23);
	    f22__ <<= (f22_ - f23_);
	    f24_ <<= sin(f24);
	    f22_3 <<= (f22__ - f24_);
	    f19_4 <<= (f19_3 + f22_3);
	    f25_ <<= sin(f25);
	    f26_ <<= sin(f26);
	    f25__ <<= (f25_ - f26_);
	    f27_ <<= sin(f27);
	    f25_3 <<= (f25__ - f27_);
	    f19_5 <<= (f19_4 + f25_3);
	    f28_ <<= sin(f28);
	    f29_ <<= sin(f29);
	    f28__ <<= (f28_ - f29_);
	    f30_ <<= sin(f30);
	    f28_3 <<= (f28__ - f30_);
	    f31_ <<= sin(f31);
	    f32_ <<= sin(f32);
	    f31__ <<= (f31_ - f32_);
	    f33_ <<= sin(f33);
	    f31_3 <<= (f31__ - f33_);
	    f28_4 <<= (f28_3 + f31_3);
	    f34_ <<= sin(f34);
	    f35_ <<= sin(f35);
	    f34__ <<= (f34_ - f35_);
	    f36_ <<= sin(f36);
	    f34_3 <<= (f34__ - f36_);
	    f28_5 <<= (f28_4 + f34_3);
	    f19_6 <<= (f19_5 * f28_5);
	    result <<= (f01_6 / f19_6),
	    "71-loop");

  return 0;
}
