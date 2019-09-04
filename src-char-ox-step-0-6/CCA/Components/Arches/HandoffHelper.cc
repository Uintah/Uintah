#include <CCA/Components/Arches/HandoffHelper.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;
using namespace ArchesCore;

void HandoffHelper::readInputFile( std::string file_name, const int index, FFInfo& info ){

  std::string varname="null";
  IntVector relative_ijk(0,0,0);
  Vector relative_xyz(0,0,0);
  // double default_value=0.;
  CellToValue values;

  gzFile file = gzopen( file_name.c_str(), "r" );
  if ( file == nullptr ) {
    throw ProblemSetupException( "Unable to open the given handoff input file: "
                                 + file_name, __FILE__, __LINE__);
  }

  info.name = getString( file );

  info.dx = getDouble( file );
  info.dy = getDouble( file );

  int num_points = getInt( file );

  int is=0; int js=0; int ks = 0;
  bool not_a_vector = false;
  if ( index == 0 ){
      is = 1;
  } else if ( index == 1 ){
      js = 1;
  } else if ( index == 2 ){
      ks = 1;
  } else {
    not_a_vector = true;
  }

  for ( int i = 0; i < num_points; i++ ) {

    int I = getInt( file );
    int J = getInt( file );
    int K = getInt( file );

    Vector v;
    double scalar;
    if ( not_a_vector ){

      scalar = getDouble( file );

      IntVector C(I,J,K);
      values.insert( std::make_pair(C,scalar));

    } else {

      v[0] = getDouble( file );
      v[1] = getDouble( file );
      v[2] = getDouble( file );
      IntVector C(I,J,K);

      values.insert( std::make_pair( C, v[index] ));

      IntVector C2(I-is, J-js, K-ks);

      values.insert( std::make_pair( C2, v[index] ));

    }
  }

  gzclose( file );

  info.values = values;

}

void HandoffHelper::readInputFile( std::string file_name, FFInfo& info ){

  std::string varname="null";
  IntVector relative_ijk(0,0,0);
  Vector relative_xyz(0,0,0);
  // double default_value=0.;
  CellToVector values;

  gzFile file = gzopen( file_name.c_str(), "r" );
  if ( file == nullptr ) {
    throw ProblemSetupException( "Unable to open the given handoff input file: "
                                 + file_name, __FILE__, __LINE__);
  }

  info.name = getString( file );

  info.dx = getDouble( file );
  info.dy = getDouble( file );

  int num_points = getInt( file );

  for ( int i = 0; i < num_points; i++ ) {

    int I = getInt( file );
    int J = getInt( file );
    int K = getInt( file );

    Vector v;

    v[0] = getDouble( file );
    v[1] = getDouble( file );
    v[2] = getDouble( file );
    IntVector C(I,J,K);

    values.insert( std::make_pair( C, v ));

  }

  gzclose( file );

  info.vec_values = values;

}
