/*
 *  Dbd.h:
 *
 *  Written by:
 *   Jason V. Morgan
 *   December 27, 2000
 *
 *  This is the base class for the DataBase Drivers for accessing 
 *  external databases.
 */
 
#ifndef Morgan_Dbi_Dbd_h_
#define Morgan_Dbi_Dbd_h_

#include <string>
#include <Packages/Morgan/share/share.h>

namespace Morgan {
namespace Dbi {

    using std::string;
    
    class Dbd {
    public:
        virtual ~Dbd();

        /* 
           bool execute
           in:  const char* statement

           Executes the given SQL statement.  Returns true if the
           operation succeeds and false if it fails (check error_str
           to find out why the call failed).  After this call, if any 
           rows are available because of a SELECT query, then at_end 
           will return false.
        */
        virtual bool execute(const char* statement) = 0;

        /*
           bool fetch
           in:  int col
           out: string& buf

           Fetches the value in column, col, from the current row
           and stores it in the given string, buf type.  If the given
           data is NULL or an error occurs, then "" is returned and 
           is_null will return true for col (fetch does not need to
           be called for is_null to return true).  If it fails, it 
           will return false, otherwise it will return true.  It will 
           fail if col < 0 or col >= number of columns.  It will also 
           fail if a row is not available (either because the last 
           executed statement was not a SELECT statement or because 
           all the current data has been read).  If an error occurs, 
           check error_str to find out what happened.
        */
        virtual bool fetch(string& buf, int col) = 0;

        /*
           bool is_null
           in:  int col

           Returns true if the data in the given column is NULL or if
           no data is available for the given column.
        */
        virtual bool is_null(int col) = 0;

        /*
           void next_row

           Fetches the next row.  If currently at the end, then this
           does nothing.  If there are no more rows, then at_end will
           return true after this call.
        */
        virtual void next_row() = 0;

        /*
           bool at_end

           Returns true if all the rows from a SELECT have been 
           read or if the last statement executed was not a SELECT
           statement.
        */
        virtual bool at_end() = 0;

        /*
           int cols

           Returns the number of columns from the current query.  This
           will return 0 if no columns or rows are available.
        */
        virtual int cols() = 0;

        /*
           void error_str
           out:  string& buf

           Copies a string for the last error into buf.  This call 
           will always copy a valid string ("OK" if no error has 
           occured).
        */
        void error_str(string& buf) const { buf = last_error; }

    protected:
        string last_error; // the last error string set
    };

    typedef Dbd* (*connect_func)(const char*, const char*, int,
                                 const char*, const char*);
} // End namespace Dbi
} // End namespace Morgan

#endif // Morgan_Dbi_Dbd_h_
