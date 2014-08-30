/*
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

#include <expression/FieldWriter.h>

#include <boost/filesystem/operations.hpp>

namespace bfs = boost::filesystem;

namespace Expr{

  //------------------------------------------------------------------

  FieldOutputDatabase::
  FieldOutputDatabase( FieldManagerList& fml,
                       const std::string dbname,
                       const bool allowOverwrite,
                       const int nEntries )
    : fml_( fml ),
      numEntries_( nEntries ),
      rootPath_( dbname ),
      allowOverwrite_( allowOverwrite ),
      isFirstWrite_( true )
  {
  }

  //------------------------------------------------------------------

  FieldOutputDatabase::
  ~FieldOutputDatabase()
  {
    for( Entries::iterator i=entries_.begin(); i!=entries_.end(); ++i ){
      delete i->second.second;
    }
  }

  //------------------------------------------------------------------

  void
  FieldOutputDatabase::
  write_database( const std::string name ) const
  {
    if( isFirstWrite_ && bfs::exists( rootPath_ ) ){
      if( !allowOverwrite_ ){
        std::ostringstream msg;
        msg << "Cannot generate an output database in directory:" << std::endl
            << "   " << bfs::system_complete(rootPath_) << std::endl
            << "Delete or rename the existing directory." << std::endl;
        throw( std::runtime_error( msg.str() ) );
      }
    }

    if( !bfs::exists( rootPath_ ) )
      bfs::create_directory( rootPath_ );

    isFirstWrite_ = false;

    bfs::create_directory( rootPath_/name );

    {
      bfs::ofstream flist( rootPath_/"databases.txt", std::ios_base::app );
      flist << name << std::endl;
    }

    bfs::ofstream mdout( rootPath_/name/"metadata.txt" );

    // write each field to a file in this subdir
    for( Entries::const_iterator ientry=entries_.begin(); ientry!=entries_.end(); ++ientry ){
      const OutputEntry& entry = ientry->second;
      const std::string& fieldname = entry.first;
      bfs::ofstream fout( rootPath_/name/fieldname );
      entry.second->write_metadata( mdout );
      entry.second->write( fout );
    }

    mdout << dblEntries_.size() << "   # number of doubles" << std::endl
          << intEntries_.size() << "   # number of ints"    << std::endl
          << strEntries_.size() << "   # number of strings" << std::endl;

    // write doubles to the file
    for( DblOutput::const_iterator idbl=dblEntries_.begin(); idbl!=dblEntries_.end(); ++idbl ){
      mdout << std::left << std::setw(20) << idbl->first << idbl->second << std::endl;
    }

    // write ints to the file
    for( IntOutput::const_iterator iint=intEntries_.begin(); iint!=intEntries_.end(); ++iint ){
      mdout << std::left << std::setw(20) << iint->first << iint->second << std::endl;
    }

    // write strings to the file
    for( StrOutput::const_iterator istr=strEntries_.begin(); istr!=strEntries_.end(); ++istr ){
      mdout << std::left << std::setw(20) << istr->first << istr->second << std::endl;
    }

//     // jcs fix this:
//     if( numEntries_ > 0 ){
//       // remove old groups if necessary.
//       fileQueue_.push( groupName );
//       if( fileQueue_.size() > size_t(numEntries_) ){
//         const std::string& nam = fileQueue_.front();
//         file.unlink( nam );
//         fileQueue_.pop();
//       }
//     }
  }

  //------------------------------------------------------------------

  void
  FieldOutputDatabase::
  extract_field_from_database( const std::string dbname,
                               const Tag fieldTag )
  {
    Entries::iterator ientry = entries_.find( fieldTag );
    // error checking...
    // ...

    OutputEntry& entry = ientry->second;
    const std::string& fieldname = entry.first;

    bfs::ifstream fin( rootPath_/dbname/fieldname );

    if( !fin.good() ){
      std::ostringstream msg;
      msg << "Could not open file " << rootPath_/dbname/fieldname << " for reading" << std::endl;
      throw std::runtime_error( msg.str() );
    }

    entry.second->read( fin );   // read the information into the field
  }

  //------------------------------------------------------------------

  void
  FieldOutputDatabase::
  request_double_output( double& var, const std::string name )
  {
    dblEntries_[name] = &var;
  }

  //------------------------------------------------------------------

  void
  FieldOutputDatabase::
  request_int_output( int& var, const std::string name )
  {
    intEntries_[name] = &var;
  }

  //------------------------------------------------------------------

  void
  FieldOutputDatabase::
  request_string_output( std::string& var, const std::string name )
  {
    strEntries_[name] = &var;
  }

  //------------------------------------------------------------------

} // namespace Expr
