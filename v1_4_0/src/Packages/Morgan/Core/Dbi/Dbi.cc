/*
 *  Dbi.cc:
 *
 *  Written by:
 *   Jason V. Morgan
 *   December 27, 2000
 *
 */

#include <vector>
#include <Packages/Morgan/Core/Dbi/Dbd.h>
#include <Packages/Morgan/share/share.h>

// headers for Dbds
#include <Packages/Morgan/Core/Dbi/ProcDbd.h>

namespace Morgan {
namespace Dbi {

using std::vector;

typedef vector<connect_func> connect_funcs_t;

static bool initialized = false; // true if the DBI has been initialized
static connect_funcs_t connect_funcs; // connection functions

static void initialize(); // initialized the DBI interface
static void add_dbds(); // add DataBase Drivers

/*
   connect
   in:  const char* database,
        const char* hostname,
        int port,
        const char* username,
        const char* password

   This function tries to connect to the given database.  It requires
   a database name at the minimum.  It can also take a hostname and a
   port number if the database is on a different system.  Hostname can
   be left blank or set to NULL if this is unnecessary info.  If a
   username is required, then a username can be passed.  Otherwise,
   the username may be left blank or set to NULL.  The same goes for
   the password.  If one of the drivers can set up a connection given
   the information passed, then a valid Dbd pointer will be returned,
   which the caller is expected to delete after it has been used.
   Otherwise NULL will be returned.
*/
Dbd* connect(const char* database, const char* hostname, int port,
             const char* username, const char* password) {
    if(!initialized) {
        initialized = true;
        initialize();
    }

    connect_funcs_t::iterator i;

    for(i = connect_funcs.begin() ; i != connect_funcs.end() ; ++i) {
        Dbd* dbd = (*i)(database, hostname, port, username, password);
        // check to see if a connection could be made
        if(dbd) {
            return dbd;
        }
    }

    return 0; // no connection could be made
}


void initialize() {
    add_dbds();
}

void add_dbds() {
    connect_funcs.push_back(ProcDbd::connect);
}

} // End namespace Dbi
} // End namespace Morgan


