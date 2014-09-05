
#include <avtudaReaderMTMDOptions.h>

#include <DBOptionsAttributes.h>

#include <string>


DBOptionsAttributes *
GetudaReaderMTMDReadOptions(void)
{
    DBOptionsAttributes *rv = new DBOptionsAttributes;
    rv->SetBool("Load extra cells", false);
    return rv;
}

DBOptionsAttributes *
GetudaReaderMTMDWriteOptions(void)
{
    DBOptionsAttributes *rv = new DBOptionsAttributes;
    return rv;
}
