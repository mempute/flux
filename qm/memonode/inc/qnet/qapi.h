
#include "sql/sqlinf.h"

#define QRYNET_PORT		3800
#define ENV_QNET_PORT	"QNET_PORT"

#define OVER_PACKING	-1

#define QRYNET_OPEN_CURSOR	0
#define QRYNET_CLOSE_CURSOR	1
#define QRYNET_REM_FEC		2 //k371so_15.
#define QRYNET_EXEC_SQL		3 //k371so_15.
#define QRYNET_PREP_STMT	4
#define QRYNET_PREP_QUERY	5
#define QRYNET_ELOB_READ	6 //k361so.begin
#define QRYNET_ELOB_WRITE	7 //end
#define QRYNET_RD_BLOB		8
#define QRYNET_WR_BLOB		9
#define QRYNET_CLOSE		10
#define QRYNET_COMMAND		11
#define QRYNET_OFF_JOURNAL	12
#define QRYNET_LOG_SQL		13
#define QRYNET_SET_QSRV_TMOUT	14
#define QRYNET_PING			15
#define QRYNET_DISABLE_CSF	16
#define QRYNET_TWIN_CONNECT	17
#define QRYNET_PULL_REPOSITORY	18
#define QRYNET_PULL_TRSLOG		19
#define QRYNET_PUSH_TRSLOG		20
#define QRYNET_PUSH_TWINSQL		21
#define QRYNET_NOTI_PRE_ACTIVE	22
#define QRYNET_NOTI_YOU_ACTIVE	23
#define QRYNET_ARE_YOU_ACTIVE   24
#define QRYNET_REPORT_IDX_ACTIVE 25
#define QRYNET_DERIVED_REPORT_IDX_ACTIVE 26
#define QRYNET_OVERWRITE_TWIN_SAVE_GATE	27	//depricate
#define QRYNET_REQ_TAB	28
#define QRYNET_WRITE_ARRAY	29 //k526so.1.
//k75so.
#define QRYSUBNET_OFF_AUTO_COMMIT	0
#define QRYSUBNET_ON_AUTO_COMMIT	1
#define QRYSUBNET_COMMIT			2
#define QRYSUBNET_ROLLBACK			3
#define QRYSUBNET_SNAPSHOT			4 //k267so.

#define QNET_OPT_NORMAL			0X00
#define QNET_OPT_UNLIMIT		0X01
#define QNET_OPT_ETERNAL		0X02
#define QNET_OPT_RESESSION		0X04

#define IS_OPT_UNLIMIT(if_connect)		QNET_OPT_UNLIMIT & if_connect
#define IS_OPT_LIMIT(if_connect)		!(IS_OPT_UNLIMIT(if_connect))
#define IS_OPT_ETERNAL(if_connect)		QNET_OPT_ETERNAL & if_connect
#define IS_OPT_TMOUT(if_connect)		!(IS_OPT_ETERNAL(if_connect))
#define IS_OPT_RESESSION(if_connect)	QNET_OPT_RESESSION & if_connect
#define IS_OPT_CLEARSESS(if_connect)	!(IS_OPT_RESESSION(if_connect))

#define PING_RV_ALIVE		0
#define PING_RV_DEAD		-1
#define PING_RV_TWIN_DEAD	-2
//맨처음 IMemAlloc호출에서 페이지할당되게하기위해
#define InitSelPage(qc)	qc->qcAlcPoint = (bytex *)INST_PAGE_OUTER_SIZE
#define ResetQCInsVar(qc) {\
	ResetSelPage(qc);\
	qc->smatMaker.icolMSel = 0;\
	qc->focusBlobDbi = nullx;\
	qc->blobSerial.pageBlob = nullx;\
}
#define SelAlloc__(dtp, qc, sz, pdat) {\
	pdat = (dtp *)qc->qcAlcPoint;\
	qc->qcAlcPoint += sz;\
	if(qc->qcAlcPoint > ((bytex *)qc->qcCurPage + INST_PAGE_OUTER_SIZE)) {\
		GetSelPage(qc);\
		pdat = (dtp *)qc->qcAlcPoint;\
		qc->qcAlcPoint += sz;\
	}\
}
//할당되는 dat 주소가 unit size 정렬되도록 할당한다.
#define SelAlloc_(dtp, qc, sz, pdat) {\
	if((longx)qc->qcAlcPoint % sizeof(unit)) \
		qc->qcAlcPoint = (bytex *)ALIGN_UNIT_((longx)qc->qcAlcPoint);\
	SelAlloc__(dtp, qc, sz, pdat);\
}
#define SelAlloc(qc, sz, pdat) SelAlloc_(bytex, qc, sz, pdat)

#define CheckPageOver(off, size) (off + size) > INST_PAGE_SIZE
#define CheckPacOver(off, size, szpac) (off + size) > szpac
#define GetSizeRemain(eat_offset_page, remain_size_total, size_feed_page, eatting_size_page) {\
	eatting_size_page = size_feed_page - eat_offset_page;\
	if(eatting_size_page > remain_size_total) {\
		eatting_size_page = remain_size_total;\
		remain_size_total = 0;\
	} else remain_size_total -= eatting_size_page;\
}

#define RemainBlobPage(off) INST_PAGE_SIZE - off
#define IsMultiPart(sz)	RemainBlobPage(sizeof(InstantPage)) < sz
#define GetLeadBlob(parg) MultiPartPoint(parg) ? sizeof(InstantPage) : 0
#define dataPointMultiPart(blob_page) blob_page + sizeof(InstantPage)

#define MAX_OPEN_ONE_SESSION	20

#define InitReadBlob(bs, sz_blob, beg_blob, lead_off) {\
	bs->remainBlob = sz_blob;\
	bs->pageBlob = beg_blob;\
	bs->offBlobPage = lead_off;\
}
#define NORMAL_QNET_SESS_CLOSE	1
#define TUFF_QNET_SESS_CLOSE	2

#ifdef __cplusplus
extern "C" {
#endif	
//k360_24so.
#define ShardGetSqlStatement(ufp) ((QueryContext *)ufp)->jrnSqlStmt
#define ShardWriteErrMsg(ufp, ercd, ermsg) PutErrMsgQC((QueryContext *)ufp, nullx, ercd, ermsg)

/*********************************************** pac ************************************/
extern intx packingResult(QueryContext *qc, bytex *page);
extern void resolvePacking(QueryContext *qc, bytex *pac, intx sz_pac);
extern void beginPrepStmtArg(SqlQuery *sq);
extern intx packingPrepStmtArg(SqlQuery *sq, bytex *page, intx &off_reg_arg);
extern intx resolvePrepStmtArg(SqlQuery *sq, bytex *pac_page, intx sz_pac);
/*********************************************** qapi ************************************/
extern intx readOutBlob(BlobStruc *bs_sor, intx &read_offset_tar, intx blob_size_sor, bytex *blob_head_page_sor, 
	intx lead_offset_sor, bytex *read_buf_tar, intx buf_size_tar);
extern intx writeInBlob(QueryContext *qc, intx &write_offset_sor, bytex *write_data_sor, intx data_size_sor, intx cvt_border);
extern intx overWriteInBlob(QueryContext *qc, uintx &write_offset, bytex *write_data_sor, intx data_size_sor, intx cvt_border); //mnj@3. jni ver
extern void DLLAPI ResetSelPage(QueryContext *qc);
extern void DLLAPI GetSelPage(QueryContext *qc);
extern void DLLAPI ReleaseSelPage(QueryContext *qc);
extern intx DLLAPI InitNetwork(void);
extern bytex *DLLAPI SQIGetVersion(bytex *vtag);
extern bytex *DLLAPI SQIGetVersion2(void);
extern bytex *DLLAPI SQIBindBuffer(SqlQuery *sq, bytex *buffer);
extern SocketAlarmBoard *DLLAPI SQIPrepareSocketAlarm(volatile shortx *net_alive_display);
extern void SQIAttachSocAlarm(SocketAlarmBoard *p_sa_board, QueryContext *qc, intx tmout);
extern void SQIDetachSocAlarm(SqlQuery *sq, intx alive_session, intx con_close, intx grace_close);
extern void SQIOnSocAlarm(QueryContext *qc, intx tmout);
extern void SQIOffSocAlarm(QueryContext *qc);
extern intx DLLAPI SQINetBoundConnect(bytex *url, bytex *bound_ip, intx if_connect, intx &i_sess, uintx &id_stamp, 
									  intx non_blk, intx tmout, bytex *user, bytex *passwd);
extern intx DLLAPI SQINetConnect(bytex *url, intx i_sess, bytex *user, bytex *passwd);
extern void DLLAPI SQINetClose(QueryContext *qc, intx alive_session);
extern void DLLAPI SQINetShutdown(QueryContext *qc);
extern void DLLAPI SQINetAutoCommit(QueryContext *qc, intx on);
extern void DLLAPI SQIQueryNetCommond(QueryContext *qc, intx command);
extern void DLLAPI SQINetOffJournal(QueryContext *qc);
extern void DLLAPI SQINoResultBreak(bytex *sql_stmt);
extern intx DLLAPI SQINetOpenCursor(QueryContext *qc);
extern void SQINetCloseCursor(QueryContext *qc);
extern void SQINetReleaseQuery(SqlQuery *sq, intx alive_session, intx con_close, intx jinst);
extern void DLLAPI SQINetExecQuery(QueryContext *qc, bytex *sql_stmt, intx sql_len);
extern void DLLAPI SQINetRequestQuery(QueryContext *active_qc, bytex *sql_stmt, intx sql_len);
extern void DLLAPI SQINetReplyQuery(QueryContext *active_qc, QueryContext *qc);
extern void DLLAPI SQINetPrepareStatement(QueryContext *qc, bytex *sql);
extern void DLLAPI SQINetRequestPrepare(QueryContext *active_qc, bytex *sql);
extern void DLLAPI SQINetReplyPrepare(QueryContext *active_qc, QueryContext *qc);
extern void DLLAPI SQINetPreparedExecuteQuery(SqlQuery *sq);
extern void DLLAPI DQINetWriteBlob(QueryContext *qc, intx icol);
extern void DLLAPI DQINetReadBlob(QueryContext *qc, bytex *dbname, bytex *tabname, intx icol, seg_t_adr ablob_head);
extern void SQINetLogQuery(QueryContext *qc, bytex *sql_stmt, intx sz_stmt);
extern void SQINetSetServTimeout(QueryContext *qc, intx tmout);
extern intx SQINetPing(intx cfd, intx itwin, intx non_blk);
extern void DLLAPI SQIWriteBlob(QueryContext *qc, bytex *blob, intx blen);
extern void DLLAPI SQIMarshalBlobArg(SqlQuery *sq, intx param_idx);
extern intx DLLAPI SQIReadBlob(BlobStruc *bs, bytex *blob, intx blen);
extern void DLLAPI SQINetWriteBatch(SqlQuery *sq, bytex *wr_stream, intx sz); //k526so.1.
/********************************************** fetch ************************************/
//SQIRun or SQIExecQuery의 리턴을 받아 호출
extern intx DLLAPI FINextSelection(QueryContext *qc);
extern intx DLLAPI FILengthColumn(QueryContext *qc, intx icol);
extern intx DLLAPI FITypeColumn(QueryContext *qc, intx icol);
extern intx DLLAPI FINumColumn(QueryContext *qc);
extern numx DLLAPI FINumSelection(QueryContext *qc);
//모든 멤버가 포인터 변수인 스트럭쳐를 정의하여 이의 포인터로 받는다, 컬럼이 넘버타입이나 데이트타입인 것은
//문자스트링으로 변환되어야 하므로 이러한 컬럼은 FIValueColumn으로 패치한다.
extern void * DLLAPI FIPointSelection(QueryContext *qc);
extern void * DLLAPI FIPointSource(QueryContext *qc, intx itab);
extern void * DLLAPI FIValueSelection(QueryContext *qc);
extern void * DLLAPI FIPointValue(QueryContext *qc, intx icol);
extern bytex * DLLAPI FICharValue(QueryContext *qc, intx icol);
extern intx DLLAPI FIIntValue(QueryContext *qc, intx icol);
extern longx DLLAPI FILongValue(QueryContext *qc, intx icol);
extern floatx DLLAPI FIFloatValue(QueryContext *qc, intx icol);
extern doublex DLLAPI FIDoubleValue(QueryContext *qc, intx icol);
extern BlobStruc * DLLAPI FIBlobValue(QueryContext *qc, intx icol);
extern intx readOutElob(QueryContext *qc, intx &read_offset_tar, bytex *read_buf_tar, intx buf_size_tar);//mnj@4. k361so.
#ifdef __cplusplus
}
#endif	
