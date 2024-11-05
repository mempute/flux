#ifndef _H_NMI
#define _H_NMI

#include "misc/pt.h"
#include "sql/sqldef.h"

#define MNI_ERMSG_LEN	2048
#define MAX_OPEN_ONE_SESSION	20

#ifdef OPT_DLL_EXP
#define DLLAPI	WINAPI
#else 
#define DLLAPI
#endif

typedef struct {
	intx ercdPCI;
	bytex ermsgPCI[MNI_ERMSG_LEN];
} ErrPCI;
struct PCIConnection_; //mnj@1. ODBC
#define ENCODING_NAME_LEN	32
typedef struct {
	void *qmdServer;
	void *pholdDriver;
	ErrPCI errdrv;
	intx sockAlrFlag; //mnj@3. jni ver
	intx utfEncoding;
	intx PCIEncode_type;
	bytex PCIEncoding[ENCODING_NAME_LEN];
	PCIConnection_ *headConn; //mnj@1. ODBC
} PCIDriver;

struct PCIStatement_;
typedef struct PCIConnection_{
	shortx notExecMProc;
	void *sqryCon;
	struct PCIStatement_ *cursorStmt[MAX_OPEN_ONE_SESSION];
	PCIDriver *mniDriver;
	ErrPCI errcon;
	struct PCIConnection_ *ptrLeft, *ptrRight; //mnj@1. ODBC
} PCIConnection;
#define PCIConnectionErrorMessage(pcon) pcon->errcon.ermsgPCI
#define PCIConnectionErrorCode(pcon) pcon->errcon.ercdPCI

#define SQL_STMT_SIZE	60000
typedef struct PCIStatement_ {
	void *sqryStmt;
	void *qcStmt;
	void *smatStmt;
	bytex stmtBuffer[SQL_STMT_SIZE], *headSqlStmt;
	PCIConnection *pconStmt;
	ErrPCI errstmt;
	intx utfEncoding2;
	intx PCIEncode_type2;
	bytex *PCIEncoding2;
} PCIStatement;
#define PCIStatementErrorMessage(pstmt) pstmt->errstmt.ermsgPCI
#define PCIStatementErrorCode(pstmt) pstmt->errstmt.ercdPCI
#define PCIResultSet(pstmt) (SelMatrix *)pstmt->smatStmt

#define RESET_MNISTMT(pstmt) {\
	pstmt->smatStmt = nullx;\
	pstmt->errstmt.ercdPCI;\
	pstmt->utfEncoding2 = 0;\
	pstmt->PCIEncoding2 = nullx;\
}

#define FindSelName(smat, col_name, ncol, icol) {\
	for(icol = 0, ncol = smat->cntSelCol;icol < ncol; icol++) {\
		if(!strcmp(col_name, SQINameSelColumn(smat, icol))) break;\
	}\
}
#define PCICopyErr(puierr, emsg, ecd) {\
	strcpy(puierr.ermsgPCI, emsg);\
	puierr.ercdPCI = ecd;\
}

typedef PCIDriver PCIHENV;
typedef PCIConnection PCIHDBC;
typedef PCIStatement PCIHSTMT;

#ifdef __cplusplus
extern "C" {
#endif
extern void * DLLAPI SchedulerDriver(PCIDriver *pdriver);
extern intx DLLAPI PCIInitDriver(PCIDriver *pdriver, intx embed, intx sync, intx journal, intx numproc, bytex *db_home, bytex *encoding, intx encode_t);
extern tJSche DLLAPI PCIGetScheduler(PCIDriver *pdriver);//k424so.
extern void DLLAPI PCIShutdownDriver(PCIDriver *pdriver);
extern PCIConnection * DLLAPI PCIGetConnection(PCIDriver *pdriver, bytex *url, bytex *user, bytex *passwd);
extern PCIConnection * DLLAPI PCIGetConnection2(void *job, bytex *url, bytex *user, bytex *passwd, intx mproc);
extern void DLLAPI PCICloseConnection2(PCIConnection *pcon);
extern void DLLAPI PCICloseConnection(PCIConnection *pcon);
extern intx DLLAPI PCICommit(PCIConnection *pcon);
extern intx DLLAPI PCIRollback(PCIConnection *pcon);
extern intx DLLAPI PCISnapshot(PCIConnection *pcon);
extern intx DLLAPI PCISetAutoCommit(PCIConnection *pcon, intx on);
extern void DLLAPI PCIBindStatement(PCIStatement *pstmt, void *squery);
extern PCIStatement * DLLAPI PCICreateStatement(PCIConnection *pcon);
extern void DLLAPI PCICloseStatement(PCIStatement *pstmt);
extern bytex *DLLAPI PCIBindBuffer(PCIStatement *pstmt, bytex *buffer);
extern void DLLAPI PCIResetBindbuf(PCIStatement *pstmt);
extern intx DLLAPI PCIExecDirect(PCIStatement *pstmt, bytex *sql);
extern intx DLLAPI PCIIsSessionClose(PCIStatement *pstmt);
extern PCIConnection * DLLAPI PCIStmtGetConnection(PCIStatement *pstmt);
extern intx DLLAPI PCIPrepare(PCIStatement *pstmt, bytex *sql);
extern intx DLLAPI PCIExcute(PCIStatement *pstmt);
extern void DLLAPI PCIMarshalLongParameter(PCIStatement *pstmt, longx val, intx param_idx);
extern void DLLAPI PCIMarshalDoubleParameter(PCIStatement *pstmt, doublex val, intx param_idx);
extern void DLLAPI PCIMarshalBytesParameter(PCIStatement *pstmt, bytex *pval, intx len, intx param_idx);
extern void DLLAPI PCIWriteBlob(PCIStatement *pstmt, bytex *blob, intx blen);
extern void DLLAPI PCIOverWriteBlob(PCIStatement *pstmt, bytex *blob, intx blen, uintx off);  //mnj@3. jni_ver
extern void DLLAPI PCIMarshalBlobParameter(PCIStatement *pstmt, intx param_idx); //mnj@5.
extern void MarshalBlobArg(SqlQuery *sqry, intx param_idx); //mnj@5.
extern intx DLLAPI PCIReadBlob(BlobStruc *bs, bytex *blob, intx blen);
extern bytex *DLLAPI PCIColumnName(PCIStatement *pstmt, intx icol);
extern intx DLLAPI PCIColumnType(PCIStatement *pstmt, intx icol);
extern intx DLLAPI PCIColumnLength(PCIStatement *pstmt, intx icol);
extern intx DLLAPI PCIColumnCount(PCIStatement *pstmt);
extern numx DLLAPI PCISelectionCount(PCIStatement *pstmt);
extern void DLLAPI PCIRestart(PCIStatement *pstmt);
extern intx DLLAPI PCIFetch(PCIStatement *pstmt);
extern bytex * DLLAPI PCIGetBineryStream(PCIStatement *pstmt, intx icol, intx *blen);
extern bytex * DLLAPI PCIGetBineryStreamByName(PCIStatement *pstmt, bytex *col_name, intx *blen);
extern bytex * DLLAPI PCIGetString(PCIStatement *pstmt, intx icol);
extern bytex * DLLAPI PCIGetStringByName(PCIStatement *pstmt, bytex *col_name);
extern void *DLLAPI PCIGetPoint(PCIStatement *pstmt, intx icol);
extern intx DLLAPI PCIGetInt(PCIStatement *pstmt, intx icol);
extern intx DLLAPI PCIGetIntByName(PCIStatement *pstmt, bytex *col_name);
extern longx DLLAPI PCIGetLong(PCIStatement *pstmt, intx icol);
extern longx DLLAPI PCIGetLongByName(PCIStatement *pstmt, bytex *col_name);
extern floatx DLLAPI PCIGetFloat(PCIStatement *pstmt, intx icol);
extern floatx DLLAPI PCIGetFloatByName(PCIStatement *pstmt, bytex *col_name);
extern doublex DLLAPI PCIGetDouble(PCIStatement *pstmt, intx icol);
extern doublex DLLAPI PCIGetDoubleByName(PCIStatement *pstmt, bytex *col_name);
extern BlobStruc * DLLAPI PCIBlobValue(PCIStatement *pstmt, intx icol);
extern intx DLLAPI PCIElobValue(QueryContext *qc, intx icol);  //mnj@2. elob. k361so.
extern intx DLLAPI PCIReadElob(PCIStatement *pstmt, bytex *blob, intx blen); //mnj@2. k361so.
extern void DLLAPI ShardAppendResult(void *ufp, SelTuple *&firststup, PCIStatement *pstmt);

extern intx DLLAPI PCIWriteBatch(PCIStatement *pstmt, bytex *wr_stream, intx sz);
extern intx DLLAPI PCISettleSchemCursor(PCIStatement *pstmt, bytex *array_name);
extern void DLLAPI PCIDesorbSchemCursor(PCIStatement *pstmt);
extern intx DLLAPI PCIEpilSequence(PCIStatement *pstmt);
extern intx DLLAPI PCIWriteItem(PCIStatement *pstmt, sytex case_dat, void *pdat, intx szdat);

extern intx DLLAPI ArtWriteArtWriteItem(PCIStatement *pstmt, bytex *ca_seq_schem, intx n_ca_seq, void *pdat, intx szdat);

#ifdef __cplusplus
}
 #endif	

#endif