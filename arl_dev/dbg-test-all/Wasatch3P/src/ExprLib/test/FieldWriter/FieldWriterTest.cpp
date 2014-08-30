#include <expression/ExprLib.h>
#include <expression/FieldWriter.h>

#include "../TestHelper.h"

#include <spatialops/Nebo.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>

#include <boost/filesystem/operations.hpp>

namespace bfs = boost::filesystem;

//--------------------------------------------------------------------

template<typename FieldT>
void add_field( Expr::FieldManagerList& fml,
                Expr::FieldOutputDatabase& db,
                const std::string name )
{
  typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.field_manager<FieldT>();
  const Expr::Tag tag( name, Expr::STATE_NONE );
  fm.register_field( tag );
  db.request_field_output<FieldT>( tag );
}

//--------------------------------------------------------------------

template<typename FieldT>
void assign_value( Expr::FieldManagerList& fml,
                   const std::string name,
                   const double val )
{
  using SpatialOps::operator<<=;
  typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.field_manager<FieldT>();
  FieldT& field = fm.field_ref( Expr::Tag(name, Expr::STATE_NONE) );
  field <<= val;
}

//--------------------------------------------------------------------

template<typename FieldT>
bool check_value( Expr::FieldManagerList& fml,
                  const std::string name,
                  const double val )
{
  typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.field_manager<FieldT>();
  FieldT& field = fm.field_ref( Expr::Tag(name, Expr::STATE_NONE) );
  for( typename FieldT::const_iterator i=field.begin(); i!=field.end(); ++i ){
    if( *i != val ) return false;
  }
  return true;
}

//--------------------------------------------------------------------

int main()
{
  try{
    Expr::FieldManagerList fml;
    Expr::FieldOutputDatabase db( fml, "testDB", true );  // allow overwrite

    // add some fields
    add_field<SpatialOps::SVolField  >( fml, db, "vol1" );
    add_field<SpatialOps::SVolField  >( fml, db, "vol2" );
    add_field<SpatialOps::SVolField  >( fml, db, "vol3" );
    add_field<SpatialOps::SSurfXField>( fml, db, "ssx1" );
    add_field<SpatialOps::SSurfYField>( fml, db, "ssy1" );
    add_field<SpatialOps::SSurfZField>( fml, db, "ssz1" );


    Expr::ExprPatch p( 10 );
    fml.allocate_fields( p.field_info() );


    // assign some values
    assign_value<SpatialOps::SVolField  >( fml, "vol1", 1.0 );
    assign_value<SpatialOps::SVolField  >( fml, "vol2", 2.0 );
    assign_value<SpatialOps::SVolField  >( fml, "vol3", 3.0 );
    assign_value<SpatialOps::SSurfXField>( fml, "ssx1", 4.0 );
    assign_value<SpatialOps::SSurfYField>( fml, "ssy1", 5.0 );
    assign_value<SpatialOps::SSurfZField>( fml, "ssz1", 6.0 );

    // write a database
    db.write_database( "t1" );
    db.write_database( "t2" );

    // ensure that the file structure was created.
    TestHelper status(true);
    bfs::path root( "testDB" );
    status( bfs::exists(root), "testDB exists" );
    status( bfs::exists(root/"t1"), "testDB/t1 exists" );
    status( bfs::exists(root/"t2"), "testDB/t2 exists" );

    status( bfs::exists(root/"t1/vol1"), "testDB/t1/vol1 exists" );
    status( bfs::exists(root/"t1/vol2"), "testDB/t1/vol2 exists" );
    status( bfs::exists(root/"t1/vol3"), "testDB/t1/vol3 exists" );
    status( bfs::exists(root/"t1/ssx1"), "testDB/t1/ssx1 exists" );
    status( bfs::exists(root/"t1/ssy1"), "testDB/t1/ssy1 exists" );
    status( bfs::exists(root/"t1/ssz1"), "testDB/t1/ssz1 exists" );

    status( bfs::exists(root/"t2/vol1"), "testDB/t2/vol1 exists" );
    status( bfs::exists(root/"t2/vol2"), "testDB/t2/vol2 exists" );
    status( bfs::exists(root/"t2/vol3"), "testDB/t2/vol3 exists" );
    status( bfs::exists(root/"t2/ssx1"), "testDB/t2/ssx1 exists" );
    status( bfs::exists(root/"t2/ssy1"), "testDB/t2/ssy1 exists" );
    status( bfs::exists(root/"t2/ssz1"), "testDB/t2/ssz1 exists" );

    // read a field from the database
    assign_value<SpatialOps::SVolField  >( fml, "vol1", 0.0 );
    assign_value<SpatialOps::SVolField  >( fml, "vol3", 0.0 );
    assign_value<SpatialOps::SSurfYField>( fml, "ssy1", 0.0 );
    db.extract_field_from_database("t1",Expr::Tag("vol1",Expr::STATE_NONE));
    db.extract_field_from_database("t2",Expr::Tag("vol3",Expr::STATE_NONE));
    db.extract_field_from_database("t2",Expr::Tag("ssy1",Expr::STATE_NONE));
    status( check_value<SpatialOps::SVolField  >(fml,"vol1",1.0), "field read 1" );
    status( check_value<SpatialOps::SVolField  >(fml,"vol3",3.0), "field read 2" );
    status( check_value<SpatialOps::SSurfYField>(fml,"ssy1",5.0), "field read 3" );
  }
  catch( std::exception& e ){
    std::cout << e.what() << std::endl;
    return -1;
  }
}

//--------------------------------------------------------------------
