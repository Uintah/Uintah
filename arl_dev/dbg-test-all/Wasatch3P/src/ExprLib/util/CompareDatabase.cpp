#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "../test/TestHelper.h"

#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

namespace po = boost::program_options;
namespace bfs = boost::filesystem;

bool verboseOutput;

//--------------------------------------------------------------------

bool check_files( const bfs::path& p1, const bfs::path& p2,
                  const double atol=0.0, const double rtol=0.0 )
{
  TestHelper status(false);

  for( bfs::directory_iterator ip1=bfs::directory_iterator(p1), ip2=bfs::directory_iterator(p2); ip1!=bfs::directory_iterator(); ++ip1, ++ip2 ){

    bfs::ifstream f1( *ip1 ), f2( *ip2 );
    if( verboseOutput ) cout << " -> " << *ip1 << "  " << *ip2 << endl;

    if( !f1.good() || !f2.good() ) return false;

    istream_iterator<string> ic1(f1), ice, ic2(f2);
    for( ; ic1!=ice; ++ic1, ++ic2 ){
      // convert the string to a double and compare that.
      // for characters, this will not be a good comparison.
      const double d1 = atof( ic1->c_str() );
      const double d2 = atof( ic2->c_str() );
      const double aerr = std::abs( d1-d2 );
      const double rerr = aerr / std::abs(d1+atol);
      if( verboseOutput ){
        if( aerr>atol && rerr>rtol ){
          cout << "\t" << p1.c_str() << " failed comparison:\n"
              << "\t\tfirst =" << *ic1 << "\n\t\tsecond=" << *ic2
              << "\n\t\tabserr=" << aerr << ", relerr=" << rerr << endl;
        }
      }
      status( aerr<=atol || rerr<=rtol, "rtol, atol" );
    }
    status( ic1==ice && ic2==ic2, "files are same length" );
  }
  if( status.isfailed() ){
    cout << "  FAIL:" << setw(30) << left << p1.native() << endl;
    return false;
  }
  if( verboseOutput ) cout << "  PASS:"  << setw(30) << left << p1.native() << endl;
  return status.ok();
}

//--------------------------------------------------------------------

bool check_directories( const bfs::path& p1, const bfs::path& p2,
                        const double atol, const double rtol )
{
  TestHelper status(false);

  if( !bfs::is_directory(p1) ){
    cout << p1 << " is not a directory" << endl;
    status(false);
  }           
  if( !is_directory(p2) ){
    cout << p1 << " is not a directory" << endl;
    status(false);
  }

  bfs::ifstream dbtext1( p1 / "databases.txt" );
  bfs::ifstream dbtext2( p2 / "databases.txt" );

  while( !dbtext1.eof() && !dbtext2.eof() ){
    string s1, s2;
    getline( dbtext1, s1 );
    getline( dbtext2, s2 );
    if( s1 != s2 ){
      cout << "databases.txt files have different entries:\n"
          << "\t" << s1 << endl
          << "\t" << s2 << endl;
      status(false);
    }
    if( s1.empty() ) break;
    try{
      if( verboseOutput ){
        cout << "Comparing:\n"
            << "\t'" << s1 << "'" << endl
            << "\t'" << s2 << "'" << endl;
      }
      status( check_files( bfs::path(p1/s1), bfs::path(p2/s2), atol, rtol ), "file check" );
    }
    catch( std::exception& err ){
      ostringstream msg;
      msg << err.what()
          << "\nError comparing files: \n"
          << bfs::path(p1/s1) << "\n\t"
          << bfs::path(p2/s2) << "\n";
      throw std::runtime_error( msg.str() );
    }
  }

  if( status.ok() ) cout << "PASS:  ";
  else              cout << "FAIL:  ";
  cout << setw(30) << left << p1.native() << "   " << p2.native() << endl;

  return status.ok();
}

//--------------------------------------------------------------------

int main( int iarg, char* carg[] )
{
  string fn1, fn2;
  double atol, rtol;

  // parse the command line options input describing the problem
  {
    po::options_description desc("\nUsage: compare_database [db1] [db2]\n\nSupported Options:");
    desc.add_options()
      ( "help", "print this help message and exit" )
      ( "atol", po::value<double>(&atol)->default_value(1e-6), "absolute tolerance" )
      ( "rtol", po::value<double>(&rtol)->default_value(1e-5), "relative tolerance" )
      ( "verbose", "activate verbose output" );

    po::options_description hidden("Hidden options");
    hidden.add_options()
      ("db1", po::value<string>(&fn1), "file 1")
      ("db2", po::value<string>(&fn2), "file 2");

    po::positional_options_description p;
    p.add("db1", 1);
    p.add("db2", 1);

    po::options_description cmdline_options;
    cmdline_options.add(desc).add(hidden);

    po::variables_map args;
    po::store( po::command_line_parser(iarg,carg).
               options(cmdline_options).positional(p).run(), args );
    po::notify(args);

    if( args.count("help") ){
      cout << desc << endl;
      return 1;
    }

    verboseOutput = args.count("verbose");

    if( fn1.empty() || fn2.empty() ){
      cout << desc << endl;
      return -1;
    }
    cout << "Comparing:\n\t" << fn1 << "\n\t" << fn2 << endl << endl;
  }

  bfs::path f1( fn1 );
  bfs::path f2( fn2 );

  if( check_directories( f1, f2, atol, rtol ) ){
    cout << endl
         << " -------------------------------" << endl
         << "   The databases are identical  " << endl
         << " -------------------------------" << endl;
    return 0;
  }
  cout << endl
       << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
       << "  The databases are different " << endl
       << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  return -1;
}

//--------------------------------------------------------------------
