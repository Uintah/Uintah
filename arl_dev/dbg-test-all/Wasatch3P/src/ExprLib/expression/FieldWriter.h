/**
 * \file   FieldWriter.h
 * \author James C. Sutherland
 *
 * Copyright (c) 2011 The University of Utah
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
#ifndef Expr_FieldWriter_h
#define Expr_FieldWriter_h

#include <map>
#include <queue>
#include <iomanip>
#include <sstream>

#include <expression/FieldManager.h>
#include <expression/FieldManagerList.h>
#include <expression/Tag.h>
#include <spatialops/structured/IntVec.h>

#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>

namespace Expr{

  struct FieldWriterBase
  {
    FieldWriterBase(){}
    virtual ~FieldWriterBase(){}
    virtual void write( boost::filesystem::ofstream& fout ) const = 0;
    virtual void write_metadata( boost::filesystem::ofstream& fout ) const = 0;
    virtual void read( boost::filesystem::ifstream& fin ) const = 0;
  };

  template< typename FieldT >
  class FieldWriter : public FieldWriterBase
  {
    typedef typename FieldMgrSelector<FieldT>::type FM;
    FM& fm_;
    const Tag tag_;
    const std::string alias_;

  public:

    FieldWriter( typename FieldMgrSelector<FieldT>::type& fm,
                 const Tag fieldTag,
                 const std::string alias )
      : fm_( fm ), tag_( fieldTag ), alias_( alias )
    {}

    ~FieldWriter(){}

    void write_metadata( boost::filesystem::ofstream& fout ) const
    {
      const FieldT& field = fm_.field_ref(tag_);
      const SpatialOps::IntVec ngm = field.get_ghost_data().get_minus();
      const SpatialOps::IntVec ngp = field.get_ghost_data().get_plus();
      assert( ngm==ngp );

      fout << std::left << std::setw(30) << alias_
           << " : " << field.window_with_ghost().extent()
           << " : " << ngm[0]
           << std::endl;
    }

    void write( boost::filesystem::ofstream& fout ) const
    {
      write_metadata( fout );
      const FieldT& field = fm_.field_ref(tag_);
      const typename FieldT::const_iterator iflde=field.end();
      for( typename FieldT::const_iterator ifld = field.begin(); ifld!=iflde; ++ifld ){
        fout << std::setprecision(16) << *ifld << " ";
      }
      fout << std::endl;
    }

    void read( boost::filesystem::ifstream& fin ) const
    {
      std::string alias, tmp;
      char ctmp;
      int nx=0, ny=0, nz=0, nghost=0;

      // name      : [ nx,ny,nz ] : nghost
      fin >> alias >> tmp >> tmp >> nx >> ctmp >> ny >> ctmp >> nz >> ctmp >> ctmp >> nghost;
      const typename SpatialOps::IntVec dim(nx,ny,nz);

      if( nx*ny*nz <= 0 ){
        std::ostringstream msg;
        msg << "Invalid information read from file for " << tag_ << std::endl
            << "  [nx,ny,nz] = " << dim << std::endl;
        throw std::runtime_error( msg.str() );
      }

      FieldT& field = fm_.field_ref(tag_);
      if( field.window_with_ghost().extent() != dim ){
        std::ostringstream msg;
        msg << "Error reading " << tag_ << " from file.  Dimension mismatch." << std::endl
            << "  expected " << field.window_with_ghost().extent() << std::endl
            << "  found    " << dim << std::endl;
        throw std::runtime_error( msg.str() );
      }

      for( typename FieldT::iterator ifld=field.begin(); ifld!=field.end(); ++ifld ){
        fin >> *ifld;
      }
    }

  }; // class FieldWriter


  /**
   *  @class  FieldOutputDatabase
   *  @author James C. Sutherland
   *  @date   November, 2008
   *  @brief  Provides a database output for time sequences of
   *          fields and metadata.
   *
   *  A FieldOutputDatabase outputs fields as individual files in
   *  subfolders on disk.  It is meant to be used in a serial
   *  calculation, and prints ghost information in all SpatialField
   *  objects by default.
   */
  class FieldOutputDatabase
  {
  public:

    /**
     *  @brief Construct a FieldOutputDatabase object
     *
     *  @param fml - the FieldManagerList
     *
     *  @param dbname - the name of this database
     *
     *  @param allowOverwrite if TRUE then this will overwrite any
     *         existing databases with the same name.
     *
     *  @param numEntries the number of entries to retain in the
     *         database.  If numEntries>0 then at most numEntries
     *         entries will be kept in the database, with the oldest
     *         entry being replaced by the current entry on subsequent
     *         calls to write_database().  If numEntries<=0, then all
     *         entries in the database will be retained.
     */
    FieldOutputDatabase( FieldManagerList& fml,
                         const std::string dbname,
                         const bool allowOverwrite = false,
                         const int numEntries=0 );

    ~FieldOutputDatabase();

    /**
     *  Write the selected information to a group with the given name.
     */
    void write_database( const std::string name ) const;

    void extract_field_from_database( const std::string dbname,
                                      const Tag fieldTag );

    /**
     *  Request output of a single double value.  This is held as a
     *  reference, so the calling program should ensure that the
     *  memory exists for the entire lifetime of this
     *  FieldOutputDatabase object.
     */
    void request_double_output( double& var, const std::string name );

    /**
     *  Request output of a single int value.  This is held as a
     *  reference, so the calling program should ensure that the
     *  memory exists for the entire lifetime of this
     *  FieldOutputDatabase object.
     */
    void request_int_output( int& var, const std::string name );

    /**
     *  Request output of a single string value.  This is held as a
     *  reference, so the calling program should ensure that the
     *  memory exists for the entire lifetime of this
     *  FieldOutputDatabase object.
     */
    void request_string_output( std::string& var, const std::string name );

    /**
     *  Request output of a field.  If no output name is given, the
     *  tag name will be used.
     */
    template< typename FieldT >
    void request_field_output( const Tag fieldTag,
                               std::string outputName="" )
    {
      if( outputName=="" ) outputName = fieldTag.name();
      const FieldWriterBase* fw = new FieldWriter<FieldT>( fml_.field_manager<FieldT>(), fieldTag, outputName );
      entries_[ fieldTag ] = make_pair( outputName, fw );
    }

  private:
    FieldManagerList& fml_;
    const int numEntries_;
    const boost::filesystem::path rootPath_;
    const bool allowOverwrite_;
    mutable bool isFirstWrite_;

    typedef std::pair< std::string, const FieldWriterBase* > OutputEntry;
    typedef std::map< Tag, OutputEntry > Entries;
    Entries entries_;

    typedef std::map< std::string, double*      > DblOutput;
    typedef std::map< std::string, int*         > IntOutput;
    typedef std::map< std::string, std::string* > StrOutput;

    DblOutput dblEntries_;
    IntOutput intEntries_;
    StrOutput strEntries_;

    std::queue<std::string>  fileQueue_;  ///< allows for removal of datasets to keep only a specified number in the database.

  };  // class FieldOutputDatabase

} // namespace Expr

#endif // Expr_FieldWriter_h
