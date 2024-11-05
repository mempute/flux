

typedef void * PCellDefinition;

#ifndef DEF_IN_VER
#define PointCode(celldef, on) ((bytex *)celldef + on)
#define PointBlob(pbhead) (bytex *)HeadEnd(pbhead)
#define OffsetCode(celldef, pcode) ((bytex *)pcode - (bytex *)celldef)
#endif