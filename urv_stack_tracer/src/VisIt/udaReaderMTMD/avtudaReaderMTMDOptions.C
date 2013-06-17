
#include <avtudaReaderMTMDOptions.h>

#include <DBOptionsAttributes.h>

#include <string>


DBOptionsAttributes *
GetudaReaderMTMDReadOptions(void)
{
    DBOptionsAttributes *rv = new DBOptionsAttributes;
    rv->SetBool("Load extra cells", true);
    return rv;
}

DBOptionsAttributes *
GetudaReaderMTMDWriteOptions(void)
{
    DBOptionsAttributes *rv = new DBOptionsAttributes;
    return rv;
}
