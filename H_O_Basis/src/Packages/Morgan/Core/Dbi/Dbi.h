/*
 *  Dbi.h:
 *
 *  Written by:
 *   Jason V. Morgan
 *   December 27, 2000
 *
 *  This is the DataBase Interface for accessing external databases.
 */

#include <Packages/Morgan/share/share.h>

namespace Morgan {
namespace Dbi {

  class Dbd;
  Dbd* connect(const char* database, const char* hostname, 
               int port, const char* username, const char* password);

} // End namespace dbi
} // End namespace Morgan


