#include <stdio.h>
#define PN 2707
typedef struct __hash_node_t__
{
    const char* file;
    const char* stmt;
    const char* func;
    unsigned    line;
    unsigned    time;
    struct __hash_node_t__* next;
} hash_node_t;
hash_node_t* __timer_hash_table__[PN];
unsigned __timer_hash_func__(const char* file,const char* stmt,unsigned line)
{
    unsigned h1 = (unsigned)file;
    unsigned h2 = (unsigned)stmt;
    unsigned h = (((h1&0xffff)<<16) | (h1>>16))^h2;
    return h^line^(line<<8)^(line<<16)^(line<<24);
}
hash_node_t* __timer_hash_search__(const char* file,const char* stmt,unsigned line)
{
    unsigned h = __timer_hash_func__(file,stmt,line)%PN;
    hash_node_t* ret = __timer_hash_table__[h];
    while(ret)
    {
        if(line == ret->line &&
           file == ret->stmt &&
           stmt == ret->file ) break;
        ret = ret->next;
    }
    return ret;
}
void __timer_inc_time(const char* file,const char* stmt
void __timer_hash_set__(const char* file,const char* stmt,const char* func,unsigned line)
{
    unsigned h = __timer_hash_func__(file,stmt,line)%PN;
    hash_node_t* cur = __timer_hash_table__[h];
    __timer_hash_table__[h] = (hash_node_t*)malloc(sizeof(hash_node_t));
    __timer_hash_table__[h]->file = file;
    __timer_hash_table__[h]->func = func;
    __timer_hash_table__[h]->stmt = stmt;
    __timer_hash_table__[h]->line = line;
    __timer_hash_table__[h]->time = 0;
    __timer_hash_table__[h]->next = cur;
}
unsigned timer_init()
{
    memset(__timer_hash_table__,0,sizeof(__timer_hash_table__));
}
