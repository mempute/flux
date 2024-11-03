
#include "sql\sqldef.h"
// hsseon, append
/*start*/
#include "qnet\pci.h"
/*end*/

#ifdef __cplusplus
extern "C" {
#endif
#define DLLimport	_declspec(dllimport)
/*********************************************** qapi ************************************/
DLLimport void WINAPI ResetSelPage(QueryContext *qc);
DLLimport void WINAPI GetSelPage(QueryContext *qc);
DLLimport void WINAPI ReleaseSelPage(QueryContext *qc);
DLLimport intx WINAPI InitNetwork(void);
DLLimport bytex *WINAPI SQIGetVersion(bytex *vtag);
DLLimport bytex *WINAPI SQIGetVersion2(void);
DLLimport bytex *WINAPI SQIBindBuffer(SqlQuery *sq, bytex *buffer);
DLLimport intx WINAPI SQINetConnect(bytex *url, intx i_sess, bytex *user, bytex *passwd);
DLLimport void WINAPI SQINetClose(QueryContext *qc, intx alive_session);
DLLimport void WINAPI SQINetShutdown(QueryContext *qc);
DLLimport void WINAPI SQIWriteBlob(QueryContext *qc, bytex *blob, intx blen);
DLLimport void WINAPI SQIMarshalBlobArg(SqlQuery *sq, intx param_idx);
DLLimport intx WINAPI SQIReadBlob(BlobStruc *bs, bytex *blob, intx blen);
/*************************************** dqiraw ***************************************/
DLLimport intx WINAPI DQIReadBlob(QueryContext *qc, bytex *dbname, bytex *tabname, intx icol, seg_t_adr ablob_head);
DLLimport intx WINAPI DQIWriteBlob(QueryContext *qc, intx icol);
DLLimport bytex * WINAPI DQIGetModuleName(void *_ufp);
DLLimport bytex * WINAPI DQIGetEntryName(void *_ufp);
//dqiraw의 이하 함수는 서버 라이브러리로서 c로 작성하는 사용자 함수에서는 lib을 정적링크하여 사용하면 되므로 dll export할필요없으나 
//자바로 사용자 함수를 작성할 경우에는 dll으로 호출하므로 필요하다.
DLLimport intx WINAPI DQINextParameter(void *_ufp);
DLLimport void * WINAPI DQIGetUserParamValue(void *_ufp, intx iarg);
DLLimport intx WINAPI DQIGetUserParamLength(void *_ufp, intx iarg);
DLLimport intx WINAPI DQIGetUserParamCount(void *_ufp);
DLLimport PCIConnection * WINAPI DQIGetConnection(void *_ufp, bytex *url, bytex *user, bytex *passwd, intx mproc);
DLLimport void WINAPI DQICloseConnection(PCIConnection *pcon);
DLLimport void WINAPI DQIPushResult(void *_ufp, intx res_t, void *val, intx len);
DLLimport void WINAPI DQIPushResultSet(void *_ufp, SelMatrix *smat);
DLLimport void WINAPI DQIResultListing(void *_ufp, void *tar, intx len);
DLLimport void WINAPI DQIPushResultList(void *_ufp, intx type);
DLLimport void WINAPI DQIIntInResultSet(void *_ufp, intx val);
DLLimport void WINAPI DQIFloatInResultSet(void *_ufp, floatx val);
DLLimport void WINAPI DQILongInResultSet(void *_ufp, longx val);
DLLimport void WINAPI DQIDoubleInResultSet(void *_ufp, doublex val);
DLLimport void WINAPI DQICharInResultSet(void *_ufp, bytex *sval, intx len);
DLLimport void WINAPI DQISelRowResultSet(void *_ufp);
DLLimport void WINAPI DQIMakeResultSet(void *_ufp, intx sel_cnt);
DLLimport void WINAPI DQITypeInResultSet(void *_ufp, sytex col_t);
DLLimport void WINAPI DQINameSlotResultSet(void *_ufp);
DLLimport void WINAPI DQINameInResultSet(void *_ufp, intx icol, bytex *sel_col);
DLLimport void WINAPI DQIUFPWriteErrorMsg(void *_ufp, intx ercd, bytex *ermsg);
/********************************************** fetch ************************************/
//SQIRun or SQIExecQuery의 리턴을 받아 호출
DLLimport intx WINAPI FINextSelection(QueryContext *qc);
DLLimport intx WINAPI FILengthColumn(QueryContext *qc, intx icol);
DLLimport intx WINAPI FITypeColumn(QueryContext *qc, intx icol);
DLLimport intx WINAPI FINumColumn(QueryContext *qc);
DLLimport numx WINAPI FINumSelection(QueryContext *qc);
//모든 멤버가 포인터 변수인 스트럭쳐를 정의하여 이의 포인터로 받는다, 컬럼이 넘버타입이나 데이트타입인 것은
//문자스트링으로 변환되어야 하므로 이러한 컬럼은 FIValueColumn으로 패치한다.
DLLimport void * WINAPI FIPointSelection(QueryContext *qc);
DLLimport void * WINAPI FIPointSource(QueryContext *qc, intx itab);
DLLimport void * WINAPI FIValueSelection(QueryContext *qc);
DLLimport void * WINAPI FIPointValue(QueryContext *qc, intx icol);
DLLimport bytex * WINAPI FICharValue(QueryContext *qc, intx icol);
DLLimport intx WINAPI FIIntValue(QueryContext *qc, intx icol);
DLLimport longx WINAPI FILongValue(QueryContext *qc, intx icol);
DLLimport floatx WINAPI FIFloatValue(QueryContext *qc, intx icol);
DLLimport doublex WINAPI FIDoubleValue(QueryContext *qc, intx icol);
DLLimport BlobStruc * WINAPI FIBlobValue(QueryContext *qc, intx icol);
/****************************************** sqi *************************************/
DLLimport tJInst WINAPI SQIInitQuery(SqlQuery *sq, tJSche sch, tJInst job_cntl, intx sfd);
DLLimport void WINAPI SQIResetQuery(SqlQuery *sq);
//sq를 더이상 사용하지 않는 맨 마지막에 이 함수를 호출한다.
DLLimport void WINAPI SQIReleaseQuery(SqlQuery *sq, intx alive_session);
DLLimport void WINAPI SQIAutoCommit(SqlQuery *sq, intx on);
DLLimport void WINAPI SQIOffJournal(SqlQuery *sq);
DLLimport intx WINAPI SQIParse(SqlQuery *sq, bytex *sql_stmt, intx sz);
//이 다음에 일반 매개변수가 있으면 마샬한다.
DLLimport void WINAPI SQIMarshalQuery(SqlQuery *sq, PCellDefinition sql_entry, PlugDescriptor *pd);
DLLimport QueryContext * WINAPI SQIRun(SqlQuery *sq, intx monitering);
DLLimport QueryContext * WINAPI SQIExecQueryn(SqlQuery *sq, bytex *sql_stmt, intx sz, intx monitering);
//DLLimport QueryContext * WINAPI SQIExecQuery(SqlQuery *sq, bytex *sql_stmt, intx monitering);
DLLimport QueryContext * WINAPI SQIPrepareStatement(SqlQuery *sq, bytex *sql_stmt, intx sz);
DLLimport QueryContext * WINAPI SQIPreparedExecuteQuery(SqlQuery *sq);
DLLimport void WINAPI SQIRegistRearQuery(tDbe dbe, bytex *qname, void *qhandle, intx qsock, bytex *qstr, FPRearQuery qfunc);
DLLimport void WINAPI SQICancelRearQuery(tDbe dbe, bytex *qname);
DLLimport void WINAPI SQIRearQueryFunc(QueryContext *qc, RQueryIt *rquery, bytex *sql_stmt);
DLLimport bytex * WINAPI SQINameSelColumn(SelMatrix *smat, intx icol);
/****************************************** pui *************************************/
DLLimport void * WINAPI SchedulerDriver(PCIDriver *pdriver);
DLLimport intx WINAPI PCIInitDriver(PCIDriver *pdriver, intx embed, intx sync, intx journal, intx numproc, bytex *db_home, bytex *encoding, intx encode_t);
DLLimport void WINAPI PCIShutdownDriver(PCIDriver *pdriver);
DLLimport PCIConnection * WINAPI PCIGetConnection(PCIDriver *pdriver, bytex *url, bytex *user, bytex *passwd);
DLLimport PCIConnection * DLLAPI PCIGetConnection2(void *job, bytex *url, bytex *user, bytex *passwd, intx mproc);
DLLimport void DLLAPI PCICloseConnection2(PCIConnection *pcon);
DLLimport void WINAPI PCICloseConnection(PCIConnection *pcon);
DLLimport intx WINAPI PCICommit(PCIConnection *pcon);
DLLimport intx WINAPI PCIRollback(PCIConnection *pcon);
DLLimport intx WINAPI PCISetAutoCommit(PCIConnection *pcon, intx on);
DLLimport void WINAPI PCIBindStatement(PCIStatement *pstmt, void *squery);
DLLimport PCIStatement * WINAPI PCICreateStatement(PCIConnection *pcon);
DLLimport void WINAPI PCICloseStatement(PCIStatement *pstmt);
DLLimport bytex *WINAPI PCIBindBuffer(PCIStatement *pstmt, bytex *buffer);
DLLimport intx WINAPI PCIExecDirect(PCIStatement *pstmt, bytex *sql);
DLLimport intx WINAPI PCIIsSessionClose(PCIStatement *pstmt);
DLLimport PCIConnection * WINAPI PCIStmtGetConnection(PCIStatement *pstmt);
DLLimport intx WINAPI PCIPrepare(PCIStatement *pstmt, bytex *sql);
DLLimport intx WINAPI PCIExcute(PCIStatement *pstmt);
DLLimport void WINAPI PCIMarshalLongParameter(PCIStatement *pstmt, longx val, intx param_idx);
DLLimport void WINAPI PCIMarshalDoubleParameter(PCIStatement *pstmt, doublex val, intx param_idx);
DLLimport void WINAPI PCIMarshalBytesParameter(PCIStatement *pstmt, bytex *pval, intx len, intx param_idx);
DLLimport void WINAPI PCIWriteBlob(PCIStatement *pstmt, bytex *blob, intx blen);
DLLimport void WINAPI PCIMarshalBlobParameter(PCIStatement *pstmt, intx param_idx);
DLLimport intx WINAPI PCIReadBlob(BlobStruc *bs, bytex *blob, intx blen);
DLLimport bytex *WINAPI PCIColumnName(PCIStatement *pstmt, intx icol);
DLLimport intx WINAPI PCIColumnType(PCIStatement *pstmt, intx icol);
DLLimport intx WINAPI PCIColumnLength(PCIStatement *pstmt, intx icol);
DLLimport intx WINAPI PCIColumnCount(PCIStatement *pstmt);
DLLimport numx WINAPI PCISelectionCount(PCIStatement *pstmt);
DLLimport void WINAPI PCIRestart(PCIStatement *pstmt);
DLLimport intx WINAPI PCIFetch(PCIStatement *pstmt);
// hsseon change
/*start*/
//DLLimport bytex * WINAPI PCIGetBineryStream(PCIStatement *pstmt, intx icol, intx &blen);
DLLimport bytex * WINAPI PCIGetBineryStream(PCIStatement *pstmt, intx icol, intx *blen);
/*end*/
// hsseon change
/*start*/
//DLLimport bytex * WINAPI PCIGetBineryStreamByName(PCIStatement *pstmt, bytex *col_name, intx &blen);
DLLimport bytex * WINAPI PCIGetBineryStreamByName(PCIStatement *pstmt, bytex *col_name, intx *blen);
/*end*/
DLLimport bytex * WINAPI PCIGetString(PCIStatement *pstmt, intx icol);
DLLimport bytex * WINAPI PCIGetStringByName(PCIStatement *pstmt, bytex *col_name);
DLLimport void *WINAPI PCIGetPoint(PCIStatement *pstmt, intx icol);
DLLimport intx WINAPI PCIGetInt(PCIStatement *pstmt, intx icol);
DLLimport intx WINAPI PCIGetIntByName(PCIStatement *pstmt, bytex *col_name);
DLLimport longx WINAPI PCIGetLong(PCIStatement *pstmt, intx icol);
DLLimport longx WINAPI PCIGetLongByName(PCIStatement *pstmt, bytex *col_name);
DLLimport floatx WINAPI PCIGetFloat(PCIStatement *pstmt, intx icol);
DLLimport floatx WINAPI PCIGetFloatByName(PCIStatement *pstmt, bytex *col_name);
DLLimport doublex WINAPI PCIGetDouble(PCIStatement *pstmt, intx icol);
DLLimport doublex WINAPI PCIGetDoubleByName(PCIStatement *pstmt, bytex *col_name);
// hsseon change
/*start*/
//DLLimport BlobStruc * WINAPI PCIBlobValue(QueryContext *qc, intx icol);
DLLimport BlobStruc * WINAPI PCIBlobValue(PCIStatement *qc, intx icol);
DLLimport void WINAPI ShardAppendResult(void *ufp, SelTuple *&firststup, PCIStatement *pstmt)
/*end*/
#ifdef __cplusplus
}
#endif
