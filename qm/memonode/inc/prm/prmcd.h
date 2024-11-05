#ifndef _H_PRMCD
#define _H_PRMCD

typedef struct PrimeInstruction_ {
	ushortx siCodeType;
	ushortx firstOpr;
	ushortx secondOpr;
	ushortx thirdOpr;
	ushortx forthOpr;
	ushortx prmResv1;
	ushortx prmResv2;
	ushortx nextInstr;
} PrimeInstruction;

#define PointPrmInstr(celldef, onpri)	(PrimeInstruction *)PointCode(celldef, onpri)
#define PointPrmOpr(pinstr, iopr)		(&pinstr->firstOpr + iopr)
#define OffsetPrmInstr(celldef, pinstr) OffsetCode(celldef, pinstr)
#endif