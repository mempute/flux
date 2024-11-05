
#include "sql/sqldef.h"
#include "qnet/pci.h"

#ifdef __cplusplus
extern "C" {
#endif

/********************************************* sqlinf *************************************/
extern void FormattingFmtCode(bytex *date_fmt_2c, bytex *dtime_fmt_2c, bytex *time_fmt_2c, bytex *digit_fmt_2c, 
								bytex *date_fmt_2d, bytex *dtime_fmt_2d, bytex *time_fmt_2d);
extern void FormatCodeQC(QueryContext *qc);
extern tSqlCb NewSqlCBuilder(void);
extern void ReleaseSqlCBuilder(tSqlCb sqlcb);
extern tJInst NewJInstance(tJSche sch, intx id_jinst);
extern void ReleaseJInstance(tJInst ji);
extern tQns NewSqlnetServ(intx sync, intx journal_on, bytex *db_home, intx cnt_sess, bytex *verify, intx ping_test, bytex *if_name, intx num_proc);
extern void ReleaseSqlnetServ(tQns sqlserv);
extern tJSche SchedulerQnetServ(tQns qserv);
extern void StartUpQnetServ(tQns qsrv, intx port);
extern tDbe GetDbEngine(tQns qsrv);

extern tSdb FindDBInstance(tJInst ji, bytex *dbname);
extern tMdb CreateMemoryDB(tJInst ji, bytex *dbname, intx db_t);
//import_url - push import db url, 이값이 있으면 import dbi,
//writable - 이값이 1이면 import dbi일때 쓰기전송도 가능, 
//export - pushUrl이 있으면 의미없고 pushUrl이 없을때 이값이 1이면 export dbi, 
//readable - export dbi일때 이값이 1이면 읽기수신도 허용한다는 의미
extern tSdb CreateSharedDB(tJInst ji, bytex *dbname, bytex *volumn_label, bytex *import_url, intx writable, intx export_, intx readable, intx db_t, intx icache, intx dcache, sytex snapshot);
//import_url - 푸쉬 임포트 디비 주소디폴트로 주어진 값으로 임포트하고 주어지지않으면 저장된 값으로
//설정된다. 나머지 싱크정보는 저장된 값 로드
extern tPdb LoadPersistDB(QueryContext *qc, bytex *dbname, bytex *volumn_label, bytex *import_url, intx db_t, bytex *passwd);
extern void LoadOptions(QueryContext *qc);
extern tNdb ConnectDataBase(tJInst ji, bytex *dbname, bytex *import_url, intx writable);
extern tDbi CreateDatabase(QueryContext *qc, bytex *dbname, bytex *volumn_label, bytex *import_url, intx writable, 
						   intx export_, intx readable, intx db_t, bytex *passwd, intx icache, intx dcache, sytex snapshot);
extern void LatchDbaJob(QueryContext *qc, intx db_t, intx lock);
extern void CutDataBase(tJInst ji, bytex *dbname);
extern void ReleaseDataBase(tJInst ji, bytex *dbname);
extern void MarkWorkingPoint(tJInst ji, bytex *rpname);
extern void RetroWorking(tJInst ji, QueryContext *qc, bytex *rpname);
extern intx CommitWork(tJInst ji, QueryContext *qc);
extern intx FlushCommitWork(tJInst ji, QueryContext *qc);
extern intx RollbackWork(tJInst ji);
extern void PutErrMsgQC(QueryContext *qc, ExptObject *eo, intx ecd, char *fmt, ...);
extern void PutPositiveQC(QueryContext *qc, intx mcd);
/****************************************** sqi *************************************/
extern tJInst DLLAPI SQIInitQuery(SqlQuery *sq, tJSche sch, tJInst job_cntl, intx sfd);
extern void DLLAPI SQIResetQuery(SqlQuery *sq);
//sq를 더이상 사용하지 않는 맨 마지막에 이 함수를 호출한다.
extern void DLLAPI SQIReleaseQuery(SqlQuery *sq, intx alive_session);
extern void DLLAPI SQIReleaseRestrict(SqlQuery *sq, intx alive_session);
extern void DLLAPI SQIAutoCommit(SqlQuery *sq, intx on);
extern intx DLLAPI SQICommit(SqlQuery *sq);
extern intx DLLAPI SQIRollback(SqlQuery *sq);
extern void DLLAPI SQIOffJournal(SqlQuery *sq);
extern void SQIPrepareDdlDqi(SqlQuery *sq, DirectQuery *&dqi);
extern intx SQIParse(SqlQuery *sq, bytex *sql_stmt, intx sz);
//이 다음에 일반 매개변수가 있으면 마샬한다.
extern void SQIMarshalQuery(SqlQuery *sq, PCellDefinition sql_entry, PlugDescriptor *pd);
extern QueryContext *SQIRun(SqlQuery *sq, intx monitering);
extern QueryContext * DLLAPI SQIExecQueryn(SqlQuery *sq, bytex *sql_stmt, intx sz, intx monitering);
#define SQIExecQuery(sq, sql_stmt, monitering) SQIExecQueryn(sq, sql_stmt, strlen(sql_stmt), monitering)
//extern QueryContext * DLLAPI SQIExecQuery(SqlQuery *sq, bytex *sql_stmt, intx monitering);
extern QueryContext * DLLAPI SQIPrepareStatement(SqlQuery *sq, bytex *sql_stmt, intx sz);
extern QueryContext *SQIPreparedExecuteQuery(SqlQuery *sq);
extern void SQIRegistRearQuery(tDbe dbe, bytex *qname, void *qhandle, intx qsock, bytex *qstr, FPRearQuery qfunc);
extern void SQICancelRearQuery(tDbe dbe, bytex *qname);
extern void SQIRearQueryFunc(QueryContext *qc, RQueryIt *rquery, bytex *sql_stmt);
extern bytex * DLLAPI SQINameSelColumn(SelMatrix *smat, intx icol);
extern QueryContext * DLLAPI SQIWriteBatch(SqlQuery *sq, bytex *wr_stream, intx sz); //k526so.1.
//k526so.2.begin.
extern intx DLLAPI SQINumChannelSequence(SqlQuery *sq);
extern QueryContext * DLLAPI SQISettleSchemCursor(SqlQuery *sq, bytex *array_name);
extern void DLLAPI SQIDesorbSchemCursor(SqlQuery *sq);
#define write_item_to_cs_numeric_case(pitem, case_cs, pcs) {\
	switch(case_cs) {\
	case BYTEX_TP:\
		*(bytex *)pcs = (bytex)atoi(pitem);\
		break;\
	case SHORTX_TP:\
		*(shortx *)pcs = (shortx)atoi(pitem);\
		break;\
	case FLOATX_TP:\
		if(*pitem == '\0') {/*kart81so*/\
			*(floatx *)pcs = 0.0;\
			*(floatx *)pcs = 0 / *(floatx *)pcs;\
		} else *(floatx *)pcs = (floatx)atof(pitem);\
		break;\
	case INTX_TP:\
		*(intx *)pcs = atoi(pitem);\
		break;\
	case LONGX_TP:\
		*(longx *)pcs = atol(pitem);\
		break;\
	case DOUBLEX_TP:\
		if(*pitem == '\0') {/*kart81so*/\
			*(doublex *)pcs = 0.0;\
			*(doublex *)pcs = 0 / *(doublex *)pcs;\
		} else *(doublex *)pcs = atof(pitem);\
		break;\
	}\
}
//kart81so.널값(nan)을 적재할 경우에는 이 매크로로 타입 변환되면 nan값이 정수형 타입으로 설정될 경우 이상값이 되므로 타입변환 안되게 즉 이매크로가 호출안되게 사용해야 한다.
#define write_item_case_to_cs_numeric(case_item, pitem, case_cs, t_cs, pcs) {\
	switch(case_item) {\
	case BYTEX_TP:\
		*(t_cs *)pcs = (t_cs)*(bytex *)pitem;\
		break;\
	case SHORTX_TP:\
		*(t_cs *)pcs = (t_cs)*(shortx *)pitem;\
		break;\
	case FLOATX_TP:\
		*(t_cs *)pcs = (t_cs)*(floatx *)pitem;\
		break;\
	case INTX_TP:\
		*(t_cs *)pcs = (t_cs)*(intx *)pitem;\
		break;\
	case LONGX_TP:\
		*(t_cs *)pcs = (t_cs)*(longx *)pitem;\
		break;\
	case DOUBLEX_TP:\
		*(t_cs *)pcs = (t_cs)*(doublex *)pitem;\
		break;\
	case NUMBERX_TP:\
		throwExpt(-1, "write item case to cs numeric invalid type\n");\
	case POINTX_TP:\
		write_item_to_cs_numeric_case((bytex *)pitem, case_cs, pcs);\
		break;\
	default:\
		throwExpt(-1, "write item case to cs numeric invalid type\n");\
	}\
}
#define write_item_case_to_cs_string(case_item, pitem, pcs) {\
	switch(case_item) {\
	case BYTEX_TP:\
		sprintf(pcs, "%d", *(bytex *)pitem);\
		break;\
	case SHORTX_TP:\
		sprintf(pcs, "%d", *(shortx *)pitem);\
		break;\
	case FLOATX_TP:\
		if(isnan(*(floatx *)pitem)) *pcs = '\0';/*kart81so.*/\
		else sprintf(pcs, "%f", *(floatx *)pitem);\
		break;\
	case INTX_TP:\
		sprintf(pcs, "%d", *(intx *)pitem);\
		break;\
	case LONGX_TP:\
		sprintf(pcs, "%lld", *(longx *)pitem);\
		break;\
	case DOUBLEX_TP:\
		if(isnan(*(doublex *)pitem)) *pcs = '\0';/*kart81so.*/\
		else sprintf(pcs, "%f", *(doublex *)pitem);\
		break;\
	case NUMBERX_TP:\
		throwExpt(-1, "write item case to cs string invalid type\n");\
		break;\
	case POINTX_TP:\
		strcpy(pcs, (bytex *)pitem);\
		break;\
	}\
	pcs += (strleng(pcs) +1);/*+1은 끝의 널문자, 널문자까지 적재.*/\
}
extern QueryContext * DLLAPI SQIWriteItem(SqlQuery *sq, sytex case_dat, void *pdat, intx szdat);
//k526so.2.end.
/*********************************************** sync *******************************************/
/********************************* Asyncronous API ***************************************/
extern tASync NewDBASync(void);
extern void ReleaseDBASync(tASync a_sync);
//dbname에 해당하는 디비로 푸쉬되는 트랜잭션 데이터를 리시브하는 큐를 생성한다. 
//dbname에 해당하는 디비인스턴스가 이미 잡인스턴스(j_inst)에 readable SDB로 등록되있어야한다.
extern void RegistDBASync(tASync a_sync, tJInst j_inst, bytex *dbname);
extern intx NextTrsData(tASync a_sync, intx &trs_code, seg_t_adr &arec, bytex *&prec);
/********************************* syncronous API ***************************************/
//싱크시그널이 올때까지 블럭대기
extern void WaitForSync(tSync syn_point);
extern tSync RegistDBSync(tSdb sdb, FPSyncCallBack synfunc, intx locsync, void *user_val);
extern tSync RegistTBSync(tSdb sdb, bytex *tbname, FPSyncCallBack synfunc,	intx locsync, void *user_val);
extern tSync RegistRecSync(tSdb sdb, unit arec, bytex *prec, FPSyncCallBack synfunc, intx locsync, void *user_val);
extern void WithdrawRecSync(tSdb sdb, unit arec, bytex *prec);
extern void TrsPullDatabase(tJInst ji, tSdb sdb, intx seq_idx);
extern void MigrationDatabase(tSdb sdb);
/*************************************** dqiraw ***************************************/
extern intx DLLAPI DQIWriteBlob(QueryContext *qc, intx icol);
extern intx DLLAPI DQIReadBlob(QueryContext *qc, bytex *dbname, bytex *tabname, intx icol, seg_t_adr ablob_head);
extern bytex *DQIGetModuleName(void *_ufp);
extern bytex *DQIGetEntryName(void *_ufp);
extern void DQISetUserProceduerStubName(void *_ufp, bytex *usf_name);
//사용자 함수에서 리스트 처리하여 리턴할때 다음 매개변수를 로우를 준비한다. 스칼라 처리이면 이 함수를 호출할 필요없음.
extern intx DQINextParameter(void *_ufp);
extern intx DQIIsScalarReturn(void *_ufp);
//매개변수의 갯수를 획득
extern intx DQIGetUserParamCount(void *_ufp);
//각 매개변수의 값을 매개변수 인덱스로 획득, 인덱스는 0부터 시작, 
//사용자 함수에서 리스트 처리할경우 두번째 부터는 위 함수를 먼저 호출한후 각 매개변수의 값을 이 함수로 획득
extern intx DQIGetUserParamType(void *_ufp, intx iarg);
extern void *DQIGetUserParamValue(void *_ufp, intx iarg);
//각 매개변수의 길이를 매개변수 인덱스로 획득
extern intx DQIGetUserParamLength(void *_ufp, intx iarg);
//사용자 함수에서 sql을 실행할 경우 이 함수를 호출하여 커낵션 객체를 얻고 이후 작업은 SQL API로 처리, 
//별도의 커낵션을 하지 않을 경우 _ufp다음 매개변수들은 의미없음.
extern PCIConnection *DQIGetConnection(void *_ufp, bytex *url, bytex *user, bytex *passwd, intx mproc);
//위 함수로 획득한 객체 반납.
extern void DQICloseConnection(PCIConnection *pcon);
//결과셋을 생성
extern void DQIMakeResultSet(void *_ufp, intx sel_cnt);
//결과셋 헤더를 생성
extern void DQINameSlotResultSet(void *_ufp);
//결과셋 헤더에 각 컬럼 이름을 설정.
extern void DQINameInResultSet(void *_ufp, intx icol, bytex *sel_col);
//결과셋 헤더에 각 컬럼 타입을 설정.
extern void DQITypeInResultSet(void *_ufp, sytex col_t);
//결과셋에 다음 로우를 준비, 첫번째 로우부터 호출되어야 함.
extern void DQISelRowResultSet(void *_ufp);
//이하 결과 셋에 각 타입별 값을 설정, 이 함수들을 호출하면 내부에서 자동으로 컬럼(vertical)이 증가됨.
extern void DQIIntInResultSet(void *_ufp, intx val);
extern void DQIFloatInResultSet(void *_ufp, floatx val);
extern void DQILongInResultSet(void *_ufp, longx val);
extern void DQIDoubleInResultSet(void *_ufp, doublex val);
extern void DQICharInResultSet(void *_ufp, bytex *sval, intx len);
//사용자 함수가 스칼라 처리일경우 스칼라 값을 반환
extern void DQIPushResult(void *_ufp, intx res_t, void *val, intx len);
//사용자 함수가 리스트 처리일 경우 각 로우별 결과 값을 리스팅
extern void DQIResultListing(void *_ufp, void *tar, intx len);
//사용자 함수가 리스트 처리일 경우 결과 리스트 반환
extern void DQIPushResultList(void *_ufp, intx type);
//cli측에서 execute명령이나 from절에서 호출하는 함수일경우 결과 셋을 리턴. 
//smat - result set을 생성하면 널 설정, 조회한 결과 셋을 바로 푸시하면 조회한 result set 설정
extern void DQIPushResultSet(void *_ufp, SelMatrix *smat);
extern void DQIUFPWriteErrorMsg(void *_ufp, intx ercd, bytex *ermsg);

extern void DQIIntSelIn(QueryContext *qc, intx val);
extern void DQIFloatSelIn(QueryContext *qc, floatx val);
extern void DQILongSelIn(QueryContext *qc, longx val);
extern void DQIDoubleSelIn(QueryContext *qc, doublex val);
extern void DQICharSelIn(QueryContext *qc, bytex *sval, intx len);
extern void DQIManualSelRow(QueryContext *qc);
extern void DQIManualSelection(QueryContext *qc, intx sel_cnt);
extern void DQISelTypeIn(QueryContext *qc, sytex col_t);
extern void DQIManualSNameSlot(QueryContext *qc);
extern void DQIManualSNameIn(QueryContext *qc, intx icol, bytex *sel_col);
/********************************* journal API ***************************************/
#ifdef JRN_VER
extern tJrnMgr NewJournalManager(tJSche sch);
extern void JournalSession(tJrnMgr jrnmgr, JournalHead *jhead, intx jrn_type);
extern void WriteJournalData(tJrnMgr jrnmgr, JournalHead *jhead, intx jrn_type, intx jrn_opr, bytex *jdat, intx dlen);
extern void ReadJournaling(tJrnMgr jrnmgr, JournalHead *jhead, bytex *slot, intx slen);
extern void JournalExecution(tJrnMgr jmgr, JournalHead *jhead, bytex *jblob, bytex *remote_jrn_syn);
extern void JrnLocalExecution(bytex *db_home);
extern void RequestJournal(bytex *jrn_srv_ip);
extern thrret ThrJrnReqFunc(thrarg asrv);
extern void StartUpJrnServ(tJSche sch);
#endif
/********************************* multi prosecution API ***************************************/
#ifdef MPROC_VER
extern void *DQIPrepareProsecution(intx num_proc);
extern void DQIBindProcQuery(QueryContext *qc, void *proc_hold);
extern void DQIGainProsecution(void *proc_hold);
extern void DQIReturnProsecution(void *proc_hold);
#else
#define DQIPrepareProsecution(num_proc)
#define DQIBindProcQuery(qc, proc_hold)
#define DQIGainProsecution(proc_hold)
#define DQIReturnProsecution(proc_hold)
#endif
#ifdef __cplusplus
}
#endif	

