/*
 *  ProcDbd.h:
 *
 *  Written by:
 *   Jason V. Morgan
 *   December 27, 2000
 *
 *  This is the Proccess DataBase Driver.  It runs a program called
 *  sql to try to connect to a database.
 */

#include <Packages/Morgan/Core/Dbi/Dbd.h>
#include <Packages/Morgan/share/share.h>
#include <Packages/Morgan/Core/Process/Proc.h>

namespace Morgan {
namespace Dbi {

    using Morgan::Process::Proc;

    class ProcDbd : public Dbd {
    public:
        ProcDbd (Proc* iproc);
        virtual ~ProcDbd ();

        bool execute(const char* statement);
        bool fetch(string& buf, int col);
        bool is_null(int col);
        void next_row();
        bool at_end();
        int rows();
        int cols();
        
        static Dbd* connect(const char*, const char*, int,
                            const char*, const char*);
    private:
        Proc* proc;

        bool bool_query(const char* query);
        int int_query(const char* query);
    };
} // End namespace Dbi
} // End namespace Morgan


