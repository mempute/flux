#ifndef _H_INF
#define _H_INF

#include "misc/xt.h"
#include "misc/stack.h"
#include "misc/sem.h"
#include "prm/prmcd.h"

#define AST_T_STRING		0
#define AST_T_INT			1
#define AST_T_LONG			2
#define AST_T_DOUBLE		3
#define AST_T_NULL			4
#define AST_T_ARGUMENT		5
#define AST_T_RELATION		6
#define AST_T_EXT_REL		7
#define AST_T_LOGICAL		8
#define AST_T_CALCULATE		9
#define AST_T_ASSIGN		10
#define AST_T_BUILT_IN_FUNC	11
#define AST_T_VARARG_FUNC	12

#define REL_OP_EQ		0
#define REL_OP_NEQ		1
#define REL_OP_GT		2
#define REL_OP_GET		3
#define REL_OP_LT		4
#define REL_OP_LET		5
#define ReverseRelOp(rop) {\
	if(rop > REL_OP_GET) rop -= 2;\
	else if(rop > REL_OP_NEQ) rop += 2;\
}
#define LOG_OP_AND		0
#define LOG_OP_OR		1
#define LOG_OP_XOR		2

#define CALC_OP_PLUS	0
#define CALC_OP_MUNUS	1
#define CALC_OP_MUL		2
#define CALC_OP_DIV		3

typedef void *tJSche;

#define SIZE_ARG_SLOT	160

typedef struct PlugDescriptor_ {
	bytex arguSlot[SIZE_ARG_SLOT];
	lsemx semPlug;
	intx returnWait;
	void *openingCell;
	intx offRegArg, idxPlugArg;
	intx idxEndArg;
	struct PlugDescriptor_ *ptrLeft, *ptrRight;
} PlugDescriptor;

typedef struct {
	intx szStack;
	intx headStack;
	PrimeInstruction **stackFrame;
} InstructStack;

#define RST_FRM_SIZE	512
typedef struct {
	PrimeInstruction *ipFrame[RST_FRM_SIZE];
	unit prmOprFrame[RST_FRM_SIZE], subOprFrame[RST_FRM_SIZE];
	InstructStack lIpStack, *ipStack;
	UnitStack lPrmOprStack, *prmOprStack, lSubOprStack, *subOprStack;
} SwitchParameter;
#define StackFrameInit(pstack, mstack, sframe, szfrm) {\
	pstack = &mstack;\
	InitStack(pstack, sframe, szfrm);\
}

#define PRE_MARSHAL_PLUG	-1 //k371so_9_2.
#define SetEntryProc(pd, cdef) pd->openingCell = cdef
#define ResetInputIdxParam(pd) pd->idxEndArg = PRE_MARSHAL_PLUG //k371so_9_2. 0 대신 -1로 리셋 표시
#define MarshalEntryProc(pd, cdef) {\
	SetEntryProc(pd, cdef);\
	ResetInputIdxParam(pd);\
}
struct PipeEntry_;

#ifdef __cplusplus
extern "C" {
#endif	
/******************************************** inf *****************************************/
extern tJSche NewScheduler(intx num_multi);
extern void SetGloValScheduler(tJSche sch, void *gval);
extern void ReleaseScheduler(tJSche js);
extern void InitSwitchParam(SwitchParameter *sp);
extern void ResetSwitchParam(SwitchParameter *sp);
extern void MarshalIntArg(PlugDescriptor *pd, intx val, intx param_idx);
extern void MarshalLongArg(PlugDescriptor *pd, longx val, intx param_idx);
extern void MarshalDoubleArg(PlugDescriptor *pd, doublex val, intx param_idx);
extern void MarshalPointArg(PlugDescriptor *pd, bytex *pval, intx len, intx multi_part, intx param_idx);
//extern void SQLMarshalLongArg(PlugDescriptor *pd, longx val, intx param_idx);
//extern void SQLMarshalDoubleArg(PlugDescriptor *pd, doublex val, intx param_idx);

/******************************************* pisch ****************************************/
extern PlugDescriptor *openPlug(tJSche jsch);
extern void closePlug(tJSche jsch, PlugDescriptor *pd);
extern void plugIn(struct PipeEntry_ *pentry, PlugDescriptor *pd, intx rwait);

#ifdef __cplusplus
}
#endif

#endif