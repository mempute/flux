
//-1 이하 음수 에러는 롤백 수행, 세션 종료

//-10는 fio 에러

//-700대는 fque 에러

//-880대는 통신 에러
#define SQNET_ERR_CONNECT	-800
#define SQNET_ERR_REQUEST	-801
#define SQNET_ERR_REPLY		-802
#define SQNET_ERR_REFUSE	-803

#define CNET_COM_ERROR		-900

#define EXPT_RET_NOT_CREATE_DB	-1000
#define EXPT_RET_IDENTIFIED_FAIL	-1001
#define EXPT_RET_LOCKED_DB_LOAD		-1002 //k296so.로드할려다 다른 세션의 디비 락 설정으로 실패하면 이후는 디비 로드없이 쿼리가 진행되므로 세션종료 코드 설정

#define EXPT_RET_FOWARD_ALARM_ALLOC		-2500



//워킹 데이터가 생성된 단계에서 발생하는 에러, 롤백은 수행, 크라이언트로 세션 종료하지 않는 에러로서 -20000 ~ -10000
#define EXPT_RET_SESSION_CLOSE				-10000
#define EXPT_RET_REC_REMOVE_OTHER_SESS		-11000
#define EXPT_RET_UNIQUE_VIOLATION			-12000
#define EXPT_RET_PRIMARY_VIOLATION			-12010
#define EXPT_RET_NOT_NULL_VIOLATION			-12020
#define EXPT_RET_FORIGN_REFER_VIOLATION		-12030
#define EXPT_RET_DEADLOCK_CHAIN_OCCUR		-12040
#define EXPT_RET_SEQUENCE_ERROR				-12050
#define EXPT_RET_MASSIVE_DML_ERROR			-12052

#define EXPT_RET_SCHEMA_CHANGED				-12060
#define EXPT_RET_LOCKED_DB_TRY_LOCK			-12061 //k296so.k297so.자신 세션이 참조한 디비 레퍼런스를 롤백과정르로 컷하기위한 에러코드 설정

#define EXPT_RET_NON_LOAD_DB				-12080

#define EXPT_RET_EXCEED_TMOUT_LOCK_RECORD	-12090
#define EXPT_RET_NOT_BUILD_PREP_STMT		-12100

#define EXPT_RET_CHANGE_PASSWD_FAIL			-12110
#define EXPT_RET_SELECT_PASSWD_FAIL			-12111
#define EXPT_RET_IDENTIFY_PASSWD_FAIL		-12112
#define EXPT_RET_PRIVIL_TABLE_FAIL			-12113
#define EXPT_RET_SCAN_PRIVIL_FAIL			-12114

#define EXPT_RET_JOIN_FAIL					-12200

#define EXPT_RET_WAIT_DECIDE_TWIN_ROLE		-12300 //k11so.

#define EXPT_RET_LIMIT_OVER_MEM_CAPA		-12400 //k114so.

#define EXPT_RET_UNDERFLOW_CIREFER_TOK		-12410
#define EXPT_RET_RETRY_CMEM_ALLOC_OVER		-12411
#define EXPT_RET_DML_FAIL					-12412

//#define EXPT_RET_VIRTUAL_MEM_OVERFLOW		-13000 vmem.h에 예약

#define EXPT_RET_CLUS_ROLLBACK_ERROR			-14000
#define EXPT_RET_CLUS_CONSTR_VIOLATION			-14001


#define SESSION_STATE_ERROR(ecd) ecd < EXPT_RET_SESSION_CLOSE
#define SESSION_CLOSE_ERROR(ecd) ecd < 0 && ecd > EXPT_RET_SESSION_CLOSE
#define ROLLBACK_ERROR(ecd) ecd < 0 && ecd > EXPT_RET_STABLE_ROLLBACK
//twin rear sql reflection에서 삽입문 실행시 오류로 인정하는 에러
#define TWIN_REAR_INSERT_ERROR(ecd) ecd > EXPT_RET_UNIQUE_VIOLATION || ecd < EXPT_RET_FORIGN_REFER_VIOLATION

//이하 워킹 데이터가 생성되지 않은 단계에서 발생하는 에러, 롤백도 하지 않고 크라이언트로 세션 종료도 
//하지않는  에러로서 -20000이하 러닝 에러
//depricate - 혹시 이들 에러중 dml수행시작 커밋/롤백이전에 cntRunFactDml가 증가됐는데 이들 에러코드로 리턴한다면 이를 롤백에러로 바꿔야한다.
#define EXPT_RET_STABLE_ROLLBACK			-20000
//#define EXPT_RET_NO_SELECT					-20002
#define EXPT_RET_NOT_FOUND_TAB				-20004
#define EXPT_RET_NO_UPDATE_VAL				-20005
#define EXPT_RET_TOO_MANY_RESULT			-20006
#define EXPT_RET_NOT_FIRST_SRC				-20007
#define EXPT_RET_NOT_THIS_EXE_PRIMARY		-20008
#define EXPT_RET_NOT_PRIMARY_COMBINE_EXIT	-20009
#define EXPT_CODE_PROPAGATE_FAIL			-20010
#define EXPT_CONTEXT_PROPAGATE_FAIL			-20011
#define EXPT_RET_MAKE_SELMAT_ERROR			-20012
#define EXPT_RET_FAULT_CURRENT_TAB			-20013
#define EXPT_RET_NOT_FOUND_COL				-20014

#define EXPT_RET_NOT_ACTIVE					-22000
#define EXPT_RET_REMOVE_USER_FAIL			-22010
#define EXPT_RET_NO_AUTHORITY_ROOT			-22011

//파싱 에러
#define EXPT_RET_PARSING_ERROR				-23000
#define EXPT_RET_LEXICAL_ERROR				-23001
//빌드 에러
#define EXPT_RET_BUILD_ERROR				-26000
#define EXPT_RET_NOT_FOUND_VIEW				-26001
#define EXPT_RET_NOT_FOUND_TABLE			-26100
#define EXPT_RET_NOT_FOUND_COLUMN			-26101
#define EXPT_RET_NOT_FOUND_DB				-26102
#define EXPT_RET_NOT_LOAD_DB				-26103
#define EXPT_RET_DUP_SINGLE_DB				-26104
#define EXPT_RET_AND_OR_REENT_SEQ			-26105 //k371so_38.런타임에 예외처리되나 성격상 빌드타임 에러이므로 여기 둔다.
#define EXPT_RET_NOT_ALLOW_REENT			-26106 //k371so_39.
#define EXPT_RET_FULL_OUTER_REENT			-26107

#define EXPT_RET_DDL_ERROR					-26200 //k93so.

//dcl 에러 
#define EXPT_RET_NOT_FOUND_CONSTRAINT_CTAB	-28000
#define EXPT_RET_NOT_FOUND_CONSTRAINT_RTAB	-28001
#define EXPT_RET_CONFLICT_DB_NAME			-28010
#define EXPT_RET_CREATE_INDEX_FAIL			-28011
#define EXPT_RET_CREATE_INDEX_SIZE_OVER		-28012
#define EXPT_RET_NOT_FOUND_REPOSITORY		-28013 //k362so.


//ddl 에러 - ddl작업중 에러가 발생하면 해당 dll수행 함수에서 직접 롤백을 수행하므로 
//			ddl에러가 발생하면 따로 롤백이나 세션 종료가 필요없다.
#define EXPT_RET_DDL_CONSTRAINT_COLUMN_INVALID	-30000
#define EXPT_RET_DDL_ALREADY_TAB				-30001
//이 사이에 이미 생성된 스키마 예외 리턴 코드 설정
#define EXPT_RET_DDL_ALREADY_CONSTRAINT			-300020
#define EXPT_RET_EXCLUSIVE_SCH_FAIL_DML_TAB		-300021 //k347so.
#define IS_RET_DDL_ALREADY(expt_cd) expt_cd <= EXPT_RET_DDL_ALREADY_TAB && expt_cd >= EXPT_RET_DDL_ALREADY_CONSTRAINT //k291so.
#define NOT_RET_DDL_ALREADY(expt_cd) expt_cd > EXPT_RET_DDL_ALREADY_TAB || expt_cd < EXPT_RET_DDL_ALREADY_CONSTRAINT

#define EXPT_RET_INDEXING_ERROR					-30100
#define EXPT_RET_CLUS_NOT_SUPPORT				-30200
#define EXPT_RET_CLUS_NET_CAST					-30201
#define EXPT_RET_CLUS_NOT_ALL					-30202
#define EXPT_RET_CLUS_EXEC_ERROR				-30203
#define EXPT_RET_CLUS_STAMP_ERROR				-30204
#define EXPT_RET_CLUS_SESS_OPEN_FAIL			-30205
#define EXPT_RET_CLUS_SESS_CLOSE				-30206
#define EXPT_RET_CLUS_SORCE_ATTATCH				-30207

//qry net에러
#define EXPT_RET_QRYNET_REENT_MUT			-32000
#define EXPT_RET_QRYNET_NOT_REENT			-32001
#define EXPT_RET_QRYNET_ELOB_CRE			-32002 //k361so.


#define EXPT_RET_USER_FUNC_CALL_ERROR		-40001
#define EXPT_RET_USER_FUNC_REFER_FAIL		-40002
#define EXPT_RET_USER_MODLUE_REFER_FAIL		-40003

//-60000번 범위 사용자 프로시져 에러 코드
#define EXPT_RET_NOT_PREPARED_JEXTSTUB		-60000

#define EXPT_RET_NOT_SEND_RESULT			-1000000




#define EXPT_RET_NO_SELECT					0	//k140so. 이 에러코드를 0로 바꿧으므로 0보다 작은 에러체크 코드들에서 이 코드를 제외시키는 것을 제거한다.
