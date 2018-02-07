
#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    ////////////////////////////////////////////////////////////////////////////
    struct GRTypes data_t;                 // data type structure
    data_t.VTXID_TYPE = VTXID_INT;         // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;         // graph size type
    data_t.VALUE_TYPE = VALUE_BELIEF;       // attributes type

    return 0;
}

